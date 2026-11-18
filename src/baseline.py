from pathlib import Path
import sys

repo_root = Path.cwd().parent  # if the notebook's working directory is notebooks/
sys.path.insert(0, str(repo_root))

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer
from src.chunking import get_tokenizer, json_to_df_new
from sklearn.svm import LinearSVC


def prepare_data(path, tokenizer, mlb, relevant_categories):
    chunks = json_to_df_new(path, tokenizer)
    chunks['categories'] = chunks['categories'].apply(
        lambda labels: [label for label in labels if label in relevant_categories] or ["None"]
    )
    X = chunks['text'].fillna('').astype(str)
    y = mlb.fit_transform(chunks['categories'])
    return X, y

def fit_baseline_model():
    tokenizer = get_tokenizer()

    # define categories for classifier

    relevant_categories = [
        "Cap On Liability", "Termination For Convenience", "Governing Law",
        "Anti-Assignment", "Change Of Control", "Renewal Term",
        "Exclusivity", "Non-Compete", "Notice Period To Terminate Renewal", "Liquidated Damages"
    ]

    relevant_categories = [category.casefold() for category in relevant_categories]

    project_root = Path.cwd().parent
    mlb = MultiLabelBinarizer(classes=relevant_categories + ["None"])
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), min_df=2,
                                sublinear_tf=True)

    X_train, y_train = prepare_data(project_root / "data/train.json", 
                                    tokenizer, mlb, relevant_categories)
    X_train_vec = vectorizer.fit_transform(X_train)

    svc = LinearSVC(class_weight='balanced')
    svc_classifier = OneVsRestClassifier(svc)
    svc_classifier.fit(X_train_vec, y_train)
    return svc_classifier, vectorizer