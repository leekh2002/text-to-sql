from __future__ import annotations

import re
from typing import TypedDict


class ValidationResult(TypedDict):
    ok: bool
    reason: str | None


ALLOWED_TABLES = {
    "cnu_courses",
    "course_prerequisite",
    "course_schedule",
    "department",
    "enrollment",
    "required_subject",
    "student",
    "subject",
}
BLOCKED_KEYWORDS = ["insert", "update", "delete", "drop", "alter", "truncate"]
ALLOWED_STUDENT_COLUMNS = {
    "admission_year",
    "completed_credits",
    "dept_code",
    "grade",
    "student_id",
}
ALLOWED_STUDENT_ID_PLACEHOLDERS = {
    ":student_id",
    "%(student_id)s",
    "{{student_id}}",
}
STUDENT_SCOPED_TABLES = {"student", "enrollment"}


def extract_table_references(sql: str) -> set[str]:
    matches = re.findall(r"\b(?:from|join)\s+([a-z_][a-z0-9_]*)\b", sql)
    return set(matches)


def extract_table_aliases(sql: str, table_name: str) -> set[str]:
    aliases = {table_name}
    matches = re.findall(
        rf"\b(?:from|join)\s+{re.escape(table_name)}(?:\s+as)?\s+([a-z_][a-z0-9_]*)\b",
        sql,
    )
    aliases.update(matches)
    return aliases


def find_disallowed_student_columns(sql: str, student_aliases: set[str]) -> set[str]:
    referenced_columns: set[str] = set()
    for alias in student_aliases:
        pattern = rf"\b{re.escape(alias)}\.([a-z_][a-z0-9_]*)\b"
        referenced_columns.update(re.findall(pattern, sql))
    return referenced_columns - ALLOWED_STUDENT_COLUMNS


def selects_student_data(sql: str, student_aliases: set[str]) -> bool:
    select_match = re.search(r"\bselect\b(.*?)\bfrom\b", sql, flags=re.DOTALL)
    if not select_match:
        return False

    select_clause = select_match.group(1)
    for alias in student_aliases:
        if re.search(rf"\b{re.escape(alias)}\.\*", select_clause):
            return True
        if re.search(rf"\b{re.escape(alias)}\.[a-z_][a-z0-9_]*\b", select_clause):
            return True

    return False


def has_scoped_student_filter(sql: str, table_aliases: set[str], requested_student_id: str | None) -> bool:
    allowed_rhs_patterns = [re.escape(placeholder) for placeholder in ALLOWED_STUDENT_ID_PLACEHOLDERS]
    if requested_student_id is not None:
        escaped_id = re.escape(str(requested_student_id))
        allowed_rhs_patterns.extend(
            [
                rf"'{escaped_id}'",
                rf'"{escaped_id}"',
                escaped_id,
            ]
        )

    rhs_group = "|".join(allowed_rhs_patterns)
    for alias in table_aliases:
        pattern = rf"\b{re.escape(alias)}\.student_id\s*=\s*(?:{rhs_group})\b"
        if re.search(pattern, sql):
            return True
    return False


def validate_generated_sql(sql: str, requested_student_id: str | None = None) -> ValidationResult:
    normalized = sql.strip().lower()
    sql_without_trailing_semicolon = normalized[:-1].rstrip() if normalized.endswith(";") else normalized

    if not re.match(r"^select\b", normalized):
        return {"ok": False, "reason": "SELECT 문만 허용됩니다."}

    if ";" in sql_without_trailing_semicolon:
        return {"ok": False, "reason": "여러 문장은 허용되지 않습니다."}

    for keyword in BLOCKED_KEYWORDS:
        if re.search(rf"\b{re.escape(keyword)}\b", sql_without_trailing_semicolon):
            return {"ok": False, "reason": f"금지 키워드 포함: {keyword}"}

    table_references = extract_table_references(sql_without_trailing_semicolon)
    if not table_references:
        return {"ok": False, "reason": "FROM 또는 JOIN 절이 필요합니다."}

    disallowed_tables = table_references - ALLOWED_TABLES
    if disallowed_tables:
        return {
            "ok": False,
            "reason": f"허용되지 않은 테이블 참조: {', '.join(sorted(disallowed_tables))}",
        }

    if "student" in table_references:
        student_aliases = extract_table_aliases(sql_without_trailing_semicolon, "student")

        disallowed_student_columns = find_disallowed_student_columns(sql_without_trailing_semicolon, student_aliases)
        if disallowed_student_columns:
            return {
                "ok": False,
                "reason": f"student에서 허용되지 않은 컬럼 참조: {', '.join(sorted(disallowed_student_columns))}",
            }

        if selects_student_data(sql_without_trailing_semicolon, student_aliases):
            return {"ok": False, "reason": "student 테이블은 필터 조건으로만 사용할 수 있습니다."}

    for table_name in STUDENT_SCOPED_TABLES & table_references:
        table_aliases = extract_table_aliases(sql_without_trailing_semicolon, table_name)
        if not has_scoped_student_filter(sql_without_trailing_semicolon, table_aliases, requested_student_id):
            return {
                "ok": False,
                "reason": f"{table_name}를 사용할 때는 요청된 학생의 학번으로 범위를 고정해야 합니다.",
            }

    if "limit" not in sql_without_trailing_semicolon:
        return {"ok": False, "reason": "기본 LIMIT가 필요합니다."}

    return {"ok": True, "reason": None}


if __name__ == "__main__":
    sample_sql = (
        "SELECT DISTINCT c.* "
        "FROM cnu_courses AS c "
        "JOIN enrollment AS e ON e.subject_code = c.subject_code "
        "WHERE e.student_id = 20240001 "
        "LIMIT 5;"
    )
    print(validate_generated_sql(sample_sql, requested_student_id="20240001"))
