import re
from dataclasses import dataclass, asdict
from typing import Any, Optional


# -------------------------------------------------------------------
# Data structures
# -------------------------------------------------------------------

@dataclass(frozen=True)
class RiskRule:
    rule_id: str
    category: str
    pattern: str
    points: float
    rationale: str
    flags: int = re.IGNORECASE | re.DOTALL


@dataclass
class RuleHit:
    rule_id: str
    points: float
    rationale: str
    matched_text: str | None


@dataclass
class ClauseRiskResult:
    category: str
    evidence: str
    classification_probability: float
    substantive_score: float
    risk_level: str
    rule_hits: list[dict]
    needs_confirmation: bool
    extracted_features: dict[str, Any]


# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

CATEGORY_BASE_SCORES = {
    "Cap On Liability": 25,
    "Termination For Convenience": 20,
    "Governing Law": 10,
    "Anti-Assignment": 15,
    "Change Of Control": 20,
    "Renewal Term": 15,
    "Exclusivity": 25,
    "Non-Compete": 30,
    "Notice Period To Terminate Renewal": 15,
    "Liquidated Damages": 25,
}


RISK_THRESHOLDS = {
    "medium": 35,
    "high": 70,
}


DEFAULT_POLICY = {
    "preferred_jurisdictions": {"new york"},
    "acceptable_jurisdictions": {
        "delaware",
        "california",
        "pennsylvania",
    },
    "high_risk_jurisdictions": set(),

    # Organization-specific thresholds
    "maximum_notice_days": 90,
    "preferred_notice_days": 30,
    "maximum_non_compete_months": 12,
    "maximum_exclusivity_months": 12,
    "maximum_liability_multiplier": 2.0,
}


# -------------------------------------------------------------------
# Text-based rules
# -------------------------------------------------------------------

RULES = [
    # Cap On Liability
    RiskRule(
        rule_id="LOL_NO_CAP",
        category="Cap On Liability",
        pattern=(
            r"\b(unlimited liability|"
            r"liability.{0,50}without (?:any )?(?:limit|limitation)|"
            r"no (?:limit|limitation).{0,40}liability)\b"
        ),
        points=50,
        rationale="The language suggests that liability may be unlimited.",
    ),
    RiskRule(
        rule_id="LOL_CAP_PRESENT",
        category="Cap On Liability",
        pattern=(
            r"\baggregate liability.{0,120}"
            r"(?:shall not exceed|will not exceed|is limited to)\b"
        ),
        points=-15,
        rationale="A defined aggregate liability cap appears to be present.",
    ),
    RiskRule(
        rule_id="LOL_CONSEQUENTIAL_ALLOWED",
        category="Cap On Liability",
        pattern=(
            r"\b(?:liable|liability).{0,80}"
            r"(?:consequential|indirect|special|punitive) damages\b"
        ),
        points=15,
        rationale="The clause may permit consequential or indirect damages.",
    ),
    RiskRule(
        rule_id="LOL_CONSEQUENTIAL_EXCLUDED",
        category="Cap On Liability",
        pattern=(
            r"\b(?:neither party|in no event).{0,100}"
            r"(?:liable|liability).{0,80}"
            r"(?:consequential|indirect|special|punitive) damages\b"
        ),
        points=-10,
        rationale="The clause appears to exclude certain consequential damages.",
    ),
    RiskRule(
        rule_id="LOL_BROAD_CARVEOUTS",
        category="Cap On Liability",
        pattern=(
            r"\b(?:limitation|cap).{0,100}"
            r"(?:shall not apply|does not apply|will not apply).{0,100}"
            r"(?:any breach|all obligations|any claim)\b"
        ),
        points=20,
        rationale="Broad exceptions may substantially reduce the protection of the cap.",
    ),

    # Termination For Convenience
    RiskRule(
        rule_id="TFC_PRESENT",
        category="Termination For Convenience",
        pattern=r"\bterminate.{0,60}(?:for convenience|without cause)\b",
        points=10,
        rationale="A termination-for-convenience right is present.",
    ),
    RiskRule(
        rule_id="TFC_IMMEDIATE",
        category="Termination For Convenience",
        pattern=(
            r"\b(?:immediately|at any time).{0,60}"
            r"terminate.{0,50}(?:for convenience|without cause)\b"
        ),
        points=20,
        rationale="Termination for convenience may occur immediately or at any time.",
    ),
    RiskRule(
        rule_id="TFC_COST_RECOVERY",
        category="Termination For Convenience",
        pattern=(
            r"\btermination.{0,150}"
            r"(?:payment for work performed|noncancelable commitments|"
            r"costs incurred|completed work)\b"
        ),
        points=-10,
        rationale="The clause appears to provide recovery for work or committed costs.",
    ),

    # Governing Law
    RiskRule(
        rule_id="GOV_LAW_UNCLEAR",
        category="Governing Law",
        pattern=r"\bapplicable law\b",
        points=5,
        rationale="The governing jurisdiction may not be expressly identified.",
    ),
    RiskRule(
        rule_id="GOV_EXCLUSIVE_FORUM",
        category="Governing Law",
        pattern=r"\bexclusive (?:jurisdiction|venue)\b",
        points=5,
        rationale="The contract requires an exclusive forum.",
    ),

    # Anti-Assignment
    RiskRule(
        rule_id="ASSIGN_CONSENT_REQUIRED",
        category="Anti-Assignment",
        pattern=(
            r"\b(?:shall not|may not|cannot) assign.{0,120}"
            r"without.{0,40}(?:prior )?written consent\b"
        ),
        points=20,
        rationale="Assignment requires prior written consent.",
    ),
    RiskRule(
        rule_id="ASSIGN_SOLE_DISCRETION",
        category="Anti-Assignment",
        pattern=r"\bconsent.{0,60}(?:sole|absolute) discretion\b",
        points=20,
        rationale="Consent may be withheld in the other party's sole discretion.",
    ),
    RiskRule(
        rule_id="ASSIGN_REASONABLE_CONSENT",
        category="Anti-Assignment",
        pattern=(
            r"\bnot unreasonably withheld"
            r"(?:,? conditioned)?(?:,? or delayed)?\b"
        ),
        points=-10,
        rationale="The ability to withhold assignment consent is limited.",
    ),
    RiskRule(
        rule_id="ASSIGN_AFFILIATE_EXCEPTION",
        category="Anti-Assignment",
        pattern=(
            r"\bassign.{0,120}"
            r"(?:affiliate|successor|merger|reorganization)\b"
        ),
        points=-10,
        rationale="The clause appears to allow assignment in certain restructurings.",
    ),
    RiskRule(
        rule_id="ASSIGN_CHANGE_CONTROL",
        category="Anti-Assignment",
        pattern=r"\bchange (?:in|of) control.{0,80}deemed.{0,30}assignment\b",
        points=20,
        rationale="A change of control may be treated as an assignment.",
    ),

    # Change Of Control
    RiskRule(
        rule_id="COC_TERMINATION",
        category="Change Of Control",
        pattern=r"\bchange (?:in|of) control.{0,140}terminat(?:e|ion)\b",
        points=30,
        rationale="A change of control may give rise to a termination right.",
    ),
    RiskRule(
        rule_id="COC_CONSENT",
        category="Change Of Control",
        pattern=(
            r"\bchange (?:in|of) control.{0,140}"
            r"(?:prior written consent|approval)\b"
        ),
        points=25,
        rationale="A change of control may require prior consent.",
    ),
    RiskRule(
        rule_id="COC_ACCELERATION",
        category="Change Of Control",
        pattern=(
            r"\bchange (?:in|of) control.{0,140}"
            r"(?:accelerat(?:e|ion)|immediately due|due and payable)\b"
        ),
        points=30,
        rationale="A change of control may accelerate payment obligations.",
    ),
    RiskRule(
        rule_id="COC_NOTICE_ONLY",
        category="Change Of Control",
        pattern=(
            r"\bchange (?:in|of) control.{0,100}"
            r"(?:provide|give).{0,30}(?:written )?notice\b"
        ),
        points=-5,
        rationale="The clause appears to require notice rather than consent.",
    ),

    # Renewal Term
    RiskRule(
        rule_id="RENEW_AUTOMATIC",
        category="Renewal Term",
        pattern=r"\b(?:automatically renew|automatic renewal)\b",
        points=15,
        rationale="The agreement renews automatically.",
    ),
    RiskRule(
        rule_id="RENEW_SUCCESSIVE",
        category="Renewal Term",
        pattern=(
            r"\b(?:successive|additional).{0,30}"
            r"(?:renewal )?terms?\b"
        ),
        points=15,
        rationale="The agreement may renew for successive terms.",
    ),
    RiskRule(
        rule_id="RENEW_PRICE_CHANGE",
        category="Renewal Term",
        pattern=(
            r"\brenewal.{0,150}"
            r"(?:increase|adjust|change).{0,60}"
            r"(?:price|pricing|fee|fees|rate|rates)\b"
        ),
        points=15,
        rationale="Pricing may change upon renewal.",
    ),
    RiskRule(
        rule_id="RENEW_MUTUAL_AGREEMENT",
        category="Renewal Term",
        pattern=(
            r"\brenew.{0,100}"
            r"(?:mutual written agreement|agreement of both parties)\b"
        ),
        points=-15,
        rationale="Renewal appears to require affirmative mutual agreement.",
    ),

    # Exclusivity
    RiskRule(
        rule_id="EXCL_PRESENT",
        category="Exclusivity",
        pattern=(
            r"\b(?:exclusive(?:ly)?|"
            r"sole (?:supplier|provider|distributor|vendor))\b"
        ),
        points=15,
        rationale="An exclusivity obligation appears to be present.",
    ),
    RiskRule(
        rule_id="EXCL_ALTERNATIVE_RESTRICTION",
        category="Exclusivity",
        pattern=(
            r"\bshall not.{0,80}"
            r"(?:purchase from|obtain from|engage|contract with).{0,80}"
            r"(?:other|another|third part)\b"
        ),
        points=20,
        rationale="The clause may restrict relationships with alternative providers.",
    ),
    RiskRule(
        rule_id="EXCL_POST_TERMINATION",
        category="Exclusivity",
        pattern=(
            r"\bexclusive.{0,160}"
            r"(?:after|following).{0,40}termination\b"
        ),
        points=25,
        rationale="The exclusivity restriction may continue after termination.",
    ),
    RiskRule(
        rule_id="EXCL_PERFORMANCE_CONDITION",
        category="Exclusivity",
        pattern=(
            r"\bexclusive.{0,160}"
            r"(?:minimum purchase|minimum sales|performance target)\b"
        ),
        points=-10,
        rationale="Exclusivity appears conditional on performance.",
    ),

    # Non-Compete
    RiskRule(
        rule_id="NC_PRESENT",
        category="Non-Compete",
        pattern=r"\b(?:non[- ]compete|shall not compete)\b",
        points=15,
        rationale="A non-compete restriction appears to be present.",
    ),
    RiskRule(
        rule_id="NC_INDEFINITE",
        category="Non-Compete",
        pattern=(
            r"\b(?:non[- ]compete|compete).{0,120}"
            r"(?:perpetual|indefinite|in perpetuity)\b"
        ),
        points=40,
        rationale="The non-compete restriction may be indefinite.",
    ),
    RiskRule(
        rule_id="NC_WORLDWIDE",
        category="Non-Compete",
        pattern=(
            r"\b(?:non[- ]compete|compete).{0,140}"
            r"(?:worldwide|anywhere in the world)\b"
        ),
        points=15,
        rationale="The restriction may have worldwide geographic scope.",
    ),
    RiskRule(
        rule_id="NC_PASSIVE_INVESTMENT_EXCEPTION",
        category="Non-Compete",
        pattern=(
            r"\b(?:passive investment|"
            r"less than \d+%|"
            r"not more than \d+%).{0,100}"
            r"(?:securities|stock|equity)\b"
        ),
        points=-5,
        rationale="The restriction appears to contain a passive-investment exception.",
    ),

    # Notice Period To Terminate Renewal
    RiskRule(
        rule_id="NOTICE_NARROW_WINDOW",
        category="Notice Period To Terminate Renewal",
        pattern=r"\bno earlier than.{0,100}no later than\b",
        points=20,
        rationale="Termination notice may only be given during a narrow window.",
    ),
    RiskRule(
        rule_id="NOTICE_RESTRICTIVE_METHOD",
        category="Notice Period To Terminate Renewal",
        pattern=(
            r"\bnotice.{0,120}"
            r"(?:certified mail only|registered mail only|"
            r"personal delivery only)\b"
        ),
        points=15,
        rationale="The required notice method may be unusually restrictive.",
    ),

    # Liquidated Damages
    RiskRule(
        rule_id="LD_PRESENT",
        category="Liquidated Damages",
        pattern=r"\bliquidated damages\b",
        points=10,
        rationale="The agreement contains liquidated damages.",
    ),
    RiskRule(
        rule_id="LD_CUMULATIVE",
        category="Liquidated Damages",
        pattern=(
            r"\bliquidated damages.{0,180}"
            r"in addition to.{0,100}"
            r"(?:other remedies|other damages|any other remedy)\b"
        ),
        points=20,
        rationale="Liquidated damages may be cumulative with other remedies.",
    ),
    RiskRule(
        rule_id="LD_SOLE_REMEDY",
        category="Liquidated Damages",
        pattern=(
            r"\bliquidated damages.{0,180}"
            r"(?:sole and exclusive remedy|exclusive remedy)\b"
        ),
        points=-15,
        rationale="Liquidated damages appear to be the exclusive remedy.",
    ),
    RiskRule(
        rule_id="LD_NO_LIABILITY_CAP",
        category="Liquidated Damages",
        pattern=(
            r"\bliquidated damages.{0,180}"
            r"(?:not subject to|excluded from|outside).{0,80}"
            r"(?:liability cap|limitation of liability)\b"
        ),
        points=25,
        rationale="Liquidated damages may be excluded from the liability cap.",
    ),
    RiskRule(
        rule_id="LD_CURE_PERIOD",
        category="Liquidated Damages",
        pattern=(
            r"\b(?:cure period|grace period|"
            r"\d+\s*days? to cure).{0,150}"
            r"liquidated damages\b"
        ),
        points=-10,
        rationale="A cure or grace period appears to apply.",
    ),
]


# -------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------

def risk_level(score: float) -> str:
    if score >= RISK_THRESHOLDS["high"]:
        return "High"
    if score >= RISK_THRESHOLDS["medium"]:
        return "Medium"
    return "Low"


def add_hit(
    hits: list[RuleHit],
    rule_id: str,
    points: float,
    rationale: str,
    matched_text: str | None,
) -> None:
    hits.append(
        RuleHit(
            rule_id=rule_id,
            points=points,
            rationale=rationale,
            matched_text=matched_text,
        )
    )


# -------------------------------------------------------------------
# Feature extraction
# -------------------------------------------------------------------

def extract_notice_days(text: str) -> int | None:
    patterns = [
        r"\bat least\s+(\d+)\s*days['’]?\s+(?:prior\s+)?(?:written\s+)?notice\b",
        r"\bnot less than\s+(\d+)\s*days['’]?\s+(?:prior\s+)?(?:written\s+)?notice\b",
        r"\b(\d+)\s*days['’]?\s+(?:prior|advance)\s+(?:written\s+)?notice\b",
        r"\bupon\s+(\d+)\s*days['’]?\s+(?:written\s+)?notice\b",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            return int(match.group(1))

    return None


def extract_duration_months(text: str) -> int | None:
    year_match = re.search(
        r"\b(?:period|term) of\s+(\d+)\s*years?\b",
        text,
        flags=re.IGNORECASE,
    )
    if year_match:
        return int(year_match.group(1)) * 12

    month_match = re.search(
        r"\b(?:period|term) of\s+(\d+)\s*months?\b",
        text,
        flags=re.IGNORECASE,
    )
    if month_match:
        return int(month_match.group(1))

    return None


def extract_fee_multiplier(text: str) -> float | None:
    numeric_match = re.search(
        r"\b(\d+(?:\.\d+)?)\s*times.{0,60}\bfees\b",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if numeric_match:
        return float(numeric_match.group(1))

    word_multipliers = {
        "one": 1.0,
        "two": 2.0,
        "three": 3.0,
        "four": 4.0,
        "five": 5.0,
    }

    word_match = re.search(
        r"\b(one|two|three|four|five)\s*times.{0,60}\bfees\b",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if word_match:
        return word_multipliers[word_match.group(1).lower()]

    return None


def extract_jurisdiction(text: str) -> str | None:
    patterns = [
        r"\blaws? of the state of\s+([A-Za-z ]+?)(?:[,.]|\s+without|\s+excluding|$)",
        r"\blaws? of\s+([A-Za-z ]+?)(?:[,.]|\s+without|\s+excluding|$)",
    ]

    for pattern in patterns:
        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if match:
            jurisdiction = re.sub(r"\s+", " ", match.group(1)).strip()
            return jurisdiction

    return None


def extract_features(
    category: str,
    evidence: str,
    supplied_features: dict[str, Any] | None,
) -> dict[str, Any]:
    """
    Automatically extracts simple numerical/text features, while allowing
    the caller to supply more reliable features.

    Example supplied_features:
        {
            "applies_to": "our_party",
            "duration_months": 24,
            "jurisdiction": "New York",
            "is_mutual": False,
            "has_automatic_renewal": True
        }
    """
    features: dict[str, Any] = {
        "applies_to": "unknown",
        "duration_months": None,
        "notice_days": None,
        "jurisdiction": None,
        "liability_cap_fee_multiplier": None,
        "is_mutual": None,
        "has_automatic_renewal": None,
        "continues_after_termination": None,
    }

    if category in {
        "Termination For Convenience",
        "Notice Period To Terminate Renewal",
    }:
        features["notice_days"] = extract_notice_days(evidence)

    if category in {
        "Renewal Term",
        "Exclusivity",
        "Non-Compete",
    }:
        features["duration_months"] = extract_duration_months(evidence)

    if category == "Cap On Liability":
        features["liability_cap_fee_multiplier"] = extract_fee_multiplier(
            evidence
        )

    if category == "Governing Law":
        features["jurisdiction"] = extract_jurisdiction(evidence)

    if re.search(
        r"\b(?:each party|either party|both parties|neither party|mutual)\b",
        evidence,
        flags=re.IGNORECASE,
    ):
        features["is_mutual"] = True

    if re.search(
        r"\b(?:automatically renew|automatic renewal)\b",
        evidence,
        flags=re.IGNORECASE,
    ):
        features["has_automatic_renewal"] = True

    if supplied_features:
        # Human-reviewed or separately extracted values take precedence.
        features.update(supplied_features)

    return features


# -------------------------------------------------------------------
# Structured feature rules
# -------------------------------------------------------------------

def apply_feature_rules(
    category: str,
    features: dict[str, Any],
    policy: dict[str, Any],
) -> list[RuleHit]:
    hits: list[RuleHit] = []

    applies_to = features.get("applies_to", "unknown")
    is_mutual = features.get("is_mutual")

    # Directionality matters most for these categories.
    directional_categories = {
        "Termination For Convenience",
        "Anti-Assignment",
        "Change Of Control",
        "Exclusivity",
        "Non-Compete",
        "Liquidated Damages",
    }

    if category in directional_categories:
        if applies_to == "our_party":
            add_hit(
                hits,
                "APPLIES_TO_OUR_PARTY",
                15,
                "The obligation or restriction appears to apply to our party.",
            )
        elif applies_to == "counterparty":
            add_hit(
                hits,
                "APPLIES_TO_COUNTERPARTY",
                -5,
                "The obligation or restriction appears to apply primarily to the counterparty.",
            )
        elif applies_to == "mutual" or is_mutual is True:
            add_hit(
                hits,
                "MUTUAL_PROVISION",
                -5,
                "The provision appears to apply mutually.",
            )
        else:
            add_hit(
                hits,
                "DIRECTION_UNCLEAR",
                5,
                "The affected party could not be determined reliably.",
            )

    # Cap On Liability
    if category == "Cap On Liability":
        multiplier = features.get("liability_cap_fee_multiplier")

        if multiplier is not None:
            maximum = policy["maximum_liability_multiplier"]

            if multiplier > maximum:
                add_hit(
                    hits,
                    "LOL_HIGH_MULTIPLIER",
                    20,
                    (
                        f"The liability cap is {multiplier:g} times fees, "
                        f"above the policy threshold of {maximum:g}."
                    ),
                )
            elif multiplier <= 1:
                add_hit(
                    hits,
                    "LOL_LOW_MULTIPLIER",
                    -10,
                    "The liability cap is no greater than one times fees.",
                )

    # Termination For Convenience
    elif category == "Termination For Convenience":
        notice_days = features.get("notice_days")

        if notice_days is None:
            add_hit(
                hits,
                "TFC_NOTICE_UNCLEAR",
                10,
                "The termination notice period could not be determined.",
            )
        elif notice_days < 30:
            add_hit(
                hits,
                "TFC_SHORT_NOTICE",
                15,
                f"The termination notice period is only {notice_days} days.",
            )
        elif notice_days >= 30:
            add_hit(
                hits,
                "TFC_REASONABLE_NOTICE",
                -10,
                f"The clause provides {notice_days} days' notice.",
            )

    # Governing Law
    elif category == "Governing Law":
        jurisdiction = features.get("jurisdiction")

        preferred = {
            item.lower()
            for item in policy.get("preferred_jurisdictions", set())
        }
        acceptable = {
            item.lower()
            for item in policy.get("acceptable_jurisdictions", set())
        }
        high_risk = {
            item.lower()
            for item in policy.get("high_risk_jurisdictions", set())
        }

        if jurisdiction is None:
            add_hit(
                hits,
                "GOV_JURISDICTION_UNKNOWN",
                15,
                "The governing jurisdiction could not be extracted.",
            )
        else:
            normalized = jurisdiction.lower()

            if normalized in preferred:
                add_hit(
                    hits,
                    "GOV_PREFERRED",
                    -10,
                    f"{jurisdiction} is a preferred jurisdiction.",
                )
            elif normalized in high_risk:
                add_hit(
                    hits,
                    "GOV_HIGH_RISK",
                    40,
                    f"{jurisdiction} is designated as high risk by policy.",
                )
            elif normalized not in acceptable:
                add_hit(
                    hits,
                    "GOV_UNAPPROVED",
                    25,
                    f"{jurisdiction} is not on the approved jurisdiction list.",
                )

    # Renewal Term
    elif category == "Renewal Term":
        duration = features.get("duration_months")

        if duration is None:
            add_hit(
                hits,
                "RENEW_DURATION_UNCLEAR",
                10,
                "The renewal-term duration could not be determined.",
            )
        elif duration > 12:
            add_hit(
                hits,
                "RENEW_LONG_TERM",
                20,
                f"The renewal term is {duration} months.",
            )
        elif duration <= 12:
            add_hit(
                hits,
                "RENEW_SHORT_TERM",
                -5,
                f"The renewal term is {duration} months or less.",
            )

    # Exclusivity
    elif category == "Exclusivity":
        duration = features.get("duration_months")

        if duration is None:
            add_hit(
                hits,
                "EXCL_DURATION_UNCLEAR",
                10,
                "The duration of exclusivity could not be determined.",
            )
        elif duration > policy["maximum_exclusivity_months"]:
            add_hit(
                hits,
                "EXCL_LONG_DURATION",
                20,
                f"Exclusivity lasts {duration} months.",
            )
        elif duration <= 6:
            add_hit(
                hits,
                "EXCL_SHORT_DURATION",
                -10,
                f"Exclusivity lasts only {duration} months.",
            )

        if features.get("continues_after_termination") is True:
            add_hit(
                hits,
                "EXCL_CONTINUES_AFTER_TERMINATION",
                25,
                "Exclusivity continues after contract termination.",
            )

    # Non-Compete
    elif category == "Non-Compete":
        duration = features.get("duration_months")

        if duration is None:
            add_hit(
                hits,
                "NC_DURATION_UNCLEAR",
                15,
                "The duration of the non-compete could not be determined.",
            )
        elif duration > policy["maximum_non_compete_months"]:
            add_hit(
                hits,
                "NC_LONG_DURATION",
                25,
                f"The non-compete lasts {duration} months.",
            )
        elif duration <= 6:
            add_hit(
                hits,
                "NC_SHORT_DURATION",
                -10,
                f"The non-compete lasts only {duration} months.",
            )

    # Notice Period To Terminate Renewal
    elif category == "Notice Period To Terminate Renewal":
        notice_days = features.get("notice_days")

        if notice_days is None:
            add_hit(
                hits,
                "NOTICE_UNCLEAR",
                20,
                "The renewal opt-out notice period could not be determined.",
            )
        elif notice_days > policy["maximum_notice_days"]:
            add_hit(
                hits,
                "NOTICE_OVER_POLICY_MAXIMUM",
                30,
                (
                    f"The notice period is {notice_days} days, above the "
                    f"{policy['maximum_notice_days']}-day policy threshold."
                ),
            )
        elif notice_days > 60:
            add_hit(
                hits,
                "NOTICE_61_TO_90",
                20,
                f"The renewal opt-out requires {notice_days} days' notice.",
            )
        elif notice_days > policy["preferred_notice_days"]:
            add_hit(
                hits,
                "NOTICE_31_TO_60",
                10,
                f"The renewal opt-out requires {notice_days} days' notice.",
            )
        else:
            add_hit(
                hits,
                "NOTICE_30_OR_LESS",
                -10,
                f"The renewal opt-out requires only {notice_days} days' notice.",
            )

    # Liquidated Damages
    elif category == "Liquidated Damages":
        if features.get("is_capped") is False:
            add_hit(
                hits,
                "LD_UNCAPPED",
                35,
                "No aggregate cap on liquidated damages was identified.",
            )
        elif features.get("is_capped") is True:
            add_hit(
                hits,
                "LD_CAPPED",
                -15,
                "Liquidated damages appear to have an aggregate cap.",
            )

        if features.get("has_cure_period") is False:
            add_hit(
                hits,
                "LD_NO_CURE_PERIOD",
                10,
                "Liquidated damages may accrue without a cure period.",
            )

    return hits


# -------------------------------------------------------------------
# Clause scoring
# -------------------------------------------------------------------

def score_clause(
    category: str,
    evidence: str,
    probability: float,
    features: dict[str, Any] | None,
    policy: dict[str, Any] | None,
    confirmation_threshold: float = 0.70,
) -> ClauseRiskResult:
    """
    Scores a single detected clause.

    `features` can contain values generated by another extraction step.
    The risk engine does not multiply legal risk by classifier probability.
    """
    if category not in CATEGORY_BASE_SCORES:
        raise ValueError(f"Unsupported category: {category}")

    active_policy = DEFAULT_POLICY.copy()
    if policy:
        active_policy.update(policy)

    extracted_features = extract_features(
        category=category,
        evidence=evidence,
        supplied_features=features,
    )

    score = CATEGORY_BASE_SCORES[category]
    hits: list[RuleHit] = []

    # Apply regex/text rules.
    for rule in RULES:
        if rule.category != category:
            continue

        match = re.search(rule.pattern, evidence, flags=rule.flags)

        if match:
            score += rule.points
            add_hit(
                hits,
                rule_id=rule.rule_id,
                points=rule.points,
                rationale=rule.rationale,
                matched_text=match.group(0),
            )

    # Apply structured feature rules.
    feature_hits = apply_feature_rules(
        category=category,
        features=extracted_features,
        policy=active_policy,
    )

    for hit in feature_hits:
        score += hit.points
        hits.append(hit)

    score = max(0.0, min(100.0, score))

    return ClauseRiskResult(
        category=category,
        evidence=evidence,
        classification_probability=probability,
        substantive_score=round(score, 1),
        risk_level=risk_level(score),
        rule_hits=[asdict(hit) for hit in hits],
        needs_confirmation=probability < confirmation_threshold,
        extracted_features=extracted_features,
    )