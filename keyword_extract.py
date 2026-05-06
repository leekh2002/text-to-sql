import argparse
import json
import os
from collections import OrderedDict
from pathlib import Path

import torch
from dotenv import load_dotenv
from rapidfuzz import fuzz, process
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine, URL
from transformers import AutoModelForTokenClassification, AutoTokenizer


MODEL_DIR = Path(__file__).resolve().parent / "course_custom_ner_model" / "models" / "course-custom-ner"
PRINT_LIMIT = 5
ENTITY_REFERENCE_MAP = {
    "DEPARTMENT": "DEPARTMENT",
    "CATEGORY": "CATEGORY",
    "COURSE_NAME": "SUBJECT",
}
CATEGORY: list[str] = []
SUBJECT: list[str] = []
DEPARTMENT: list[str] = []


def build_postgres_url() -> str:
    url = URL.create(
        drivername="postgresql+psycopg2",
        username=os.getenv("POSTGRES_USER", "postgres"),
        password=os.getenv("POSTGRES_PASSWORD", "postgres"),
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        database=os.getenv("POSTGRES_DB", "postgres"),
    )
    return url.render_as_string(hide_password=False)


def create_db_engine() -> Engine:
    load_dotenv()
    engine = create_engine(build_postgres_url())
    return engine


def fetch_distinct_values(engine: Engine, table: str, column: str) -> list[str]:
    sql = text(
        f"""
        SELECT DISTINCT {column}
        FROM {table}
        WHERE {column} IS NOT NULL
        ORDER BY {column}
        """
    )
    with engine.connect() as conn:
        return [row[0] for row in conn.execute(sql).fetchall() if row[0]]


def load_reference_values(engine: Engine | None = None) -> dict[str, list[str]]:
    owns_engine = engine is None
    if owns_engine:
        engine = create_db_engine()

    try:
        with engine.connect():
            pass

        category_values = fetch_distinct_values(engine, "subject", "category")
        subject_values = fetch_distinct_values(engine, "subject", "subject_name")
        department_values = fetch_distinct_values(engine, "department", "dept_name")

        CATEGORY.clear()
        CATEGORY.extend(category_values)
        SUBJECT.clear()
        SUBJECT.extend(subject_values)
        DEPARTMENT.clear()
        DEPARTMENT.extend(department_values)

        return {
            "CATEGORY": CATEGORY[:PRINT_LIMIT],
            "SUBJECT": SUBJECT[:PRINT_LIMIT],
            "DEPARTMENT": DEPARTMENT[:PRINT_LIMIT],
        }
    finally:
        if owns_engine:
            engine.dispose()


def slot_key(label: str) -> str:
    return label.lower()


class CourseNERPredictor:
    def __init__(self, model_dir: str | Path = MODEL_DIR, max_length: int = 128):
        self.model_dir = Path(model_dir)
        self.max_length = max_length
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = AutoModelForTokenClassification.from_pretrained(self.model_dir).to(self.device)
        self.model.eval()

        label_path = self.model_dir / "labels.json"
        with open(label_path, "r", encoding="utf-8") as file:
            label_data = json.load(file)

        self.id_to_label = {
            int(key): value
            for key, value in label_data["id_to_label"].items()
        }

    @torch.no_grad()
    def extract(self, text: str) -> list[dict]:
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            return_offsets_mapping=True,
            truncation=True,
            max_length=self.max_length,
        )

        offsets = encoded.pop("offset_mapping")[0].tolist()
        encoded = {
            key: value.to(self.device)
            for key, value in encoded.items()
        }

        outputs = self.model(**encoded)
        probs = torch.softmax(outputs.logits[0], dim=-1)

        pred_ids = torch.argmax(probs, dim=-1).tolist()
        pred_scores = torch.max(probs, dim=-1).values.tolist()

        token_predictions = []
        for pred_id, score, (start, end) in zip(pred_ids, pred_scores, offsets):
            if start == end:
                continue

            token_predictions.append(
                {
                    "start": start,
                    "end": end,
                    "label": self.id_to_label[pred_id],
                    "score": float(score),
                }
            )

        return self._merge_bio(text, token_predictions)

    def _merge_bio(self, text: str, token_predictions: list[dict]) -> list[dict]:
        entities = []
        current = None

        for token in token_predictions:
            label = token["label"]

            if label == "O":
                if current:
                    entities.append(current)
                    current = None
                continue

            if "-" not in label:
                continue

            prefix, entity_type = label.split("-", 1)

            if prefix == "B" or current is None or current["label"] != entity_type:
                if current:
                    entities.append(current)
                current = {
                    "label": entity_type,
                    "start": token["start"],
                    "end": token["end"],
                    "scores": [token["score"]],
                }
                continue

            current["end"] = token["end"]
            current["scores"].append(token["score"])

        if current:
            entities.append(current)

        result = []
        for entity in entities:
            start = entity["start"]
            end = entity["end"]
            avg_score = sum(entity["scores"]) / len(entity["scores"])
            result.append(
                {
                    "text": text[start:end],
                    "label": entity["label"],
                    "start": start,
                    "end": end,
                    "score": round(avg_score, 4),
                }
            )

        return result


def build_slots(entities: list[dict]) -> dict[str, list[str]]:
    slots: OrderedDict[str, list[str]] = OrderedDict()

    for entity in entities:
        key = slot_key(entity["label"])
        if key not in slots:
            slots[key] = []

        value = entity["text"]
        if value not in slots[key]:
            slots[key].append(value)

    return dict(slots)


def get_reference_values_by_label(label: str) -> list[str]:
    if label == "DEPARTMENT":
        return DEPARTMENT
    if label == "CATEGORY":
        return CATEGORY
    if label == "COURSE_NAME":
        return SUBJECT
    return []


def find_best_db_match(entity_text: str, candidates: list[str]) -> tuple[str | None, float]:
    if not entity_text or not candidates:
        return None, 0.0

    match = process.extractOne(
        entity_text,
        candidates,
        scorer=fuzz.ratio,
    )
    if match is None:
        return None, 0.0

    matched_text, score, _ = match
    return matched_text, float(score)


def correct_ner_entities(entities: list[dict]) -> list[dict]:
    corrected_entities = []

    for entity in entities:
        label = entity["label"]
        candidates = get_reference_values_by_label(label)

        if label not in ENTITY_REFERENCE_MAP or not candidates:
            corrected_entities.append(
                {
                    **entity,
                    "corrected_text": entity["text"],
                    "matched_label_pool": None,
                    "similarity": None,
                }
            )
            continue

        best_match, similarity = find_best_db_match(entity["text"], candidates)
        corrected_entities.append(
            {
                **entity,
                "corrected_text": best_match or entity["text"],
                "matched_label_pool": ENTITY_REFERENCE_MAP[label],
                "similarity": round(similarity, 2),
            }
        )

    return corrected_entities


def build_corrected_query(query: str, corrected_entities: list[dict]) -> str:
    corrected_query = query

    for entity in sorted(corrected_entities, key=lambda item: item["start"], reverse=True):
        corrected_text = entity.get("corrected_text", entity["text"])
        corrected_query = (
            corrected_query[: entity["start"]]
            + corrected_text
            + corrected_query[entity["end"] :]
        )

    return corrected_query


def extract_keywords(query: str, model_dir: str | Path = MODEL_DIR, max_length: int = 128) -> dict:
    predictor = CourseNERPredictor(model_dir=model_dir, max_length=max_length)
    entities = predictor.extract(query)
    corrected_entities = correct_ner_entities(entities)
    return {
        "query": query,
        "entities": entities,
        "slots": build_slots(entities),
        "corrected_entities": corrected_entities,
        "corrected_slots": build_slots(corrected_entities),
        "corrected_query": build_corrected_query(query, corrected_entities),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True, help="NER로 분석할 자연어 질의")
    parser.add_argument("--model_dir", default=str(MODEL_DIR))
    parser.add_argument("--max_length", type=int, default=128)
    args = parser.parse_args()

    load_reference_values()
    result = extract_keywords(
        query=args.query,
        model_dir=args.model_dir,
        max_length=args.max_length,
    )
    print(result["corrected_query"])


if __name__ == "__main__":
    main()
