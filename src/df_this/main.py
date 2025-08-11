import pandas as pd
from tqdm import tqdm
import string

def check_type(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pandas.DataFrame, got {type(df).__name__}")
    else:
        return True

def df_desc(df: pd.DataFrame) -> pd.DataFrame:
    check_type(df)

    brackets_quotes = set('\'"()[]{}')
    punctuation = set('.,;:!?')
    operators = set('+-*/=%<>')

    all_ascii = set(chr(i) for i in range(128))

    known_ascii = set(string.digits + string.ascii_letters).union(
        brackets_quotes, punctuation, operators, set(' \t\n\r')
    )

    other_ascii = all_ascii - known_ascii

    results = []

    for col in tqdm(df.columns, leave=False, disable=False):
        chars = {
            "numeric": set(),
            "letters_lowercase": set(),
            "letters_uppercase": set(),
            "whitespace": set(),
            "brackets_quotes": set(),
            "punctuation": set(),
            "operators": set(),
            "other_ascii": set(),
            "other_non_ascii": set(),
    	}

        series = df[col].dropna().astype(str)

        for val in series:
            for ch in val:
                if ch.isdigit():
                    chars["numeric"].add(ch)
                elif ch.islower():
                    chars["letters_lowercase"].add(ch)
                elif ch.isupper():
                    chars["letters_uppercase"].add(ch)
                elif ch.isspace():
                    chars["whitespace"].add(ch)
                elif ch in brackets_quotes:
                    chars["brackets_quotes"].add(ch)
                elif ch in punctuation:
                    chars["punctuation"].add(ch)
                elif ch in operators:
                    chars["operators"].add(ch)
                elif ord(ch) < 128 and ch in other_ascii:
                    chars["other_ascii"].add(ch)
                else:
                    chars["other_non_ascii"].add(ch)

        results.append({
            "column": col,
            **{group: "".join(sorted(chars[group])) for group in chars}
        })

    return pd.DataFrame(results)


def df_stats(df: pd.DataFrame) -> pd.DataFrame:
    check_type(df)

    numeric_df = df.select_dtypes(include=["number"])

    summary = []

    for col in tqdm(numeric_df.columns, leave=False, disable=False):
        values = numeric_df[col].dropna()
        summary.append({
            "column":col,
            "min":values.min(),
            "max":values.max(),
            "mean":values.mean(),
            "median":values.median(),
            "std_smaple":values.std(ddof=1),
            "std_pop": values.std(ddof=0)
        })
    
    return pd.DataFrame(summary)


def df_nullique(df: pd.DataFrame) -> pd.DataFrame:
    check_type(df)

    result = []

    for col in tqdm(df.columns, leave=False, disable=False):
        series = df[col]

        cleaned = series.replace(r'^\s*$', '', regex=True)

        is_unique = cleaned.duplicated(keep=False).sum() == 0

        distinct_values = cleaned.fillna("<<NULL>>").replace("", "<<EMPTY>>")
        distinct_count = distinct_values.nunique(dropna=False)

        has_null = cleaned.isnull().any()
        has_empty = (cleaned == "").any()

        if has_null and has_empty:
            null_type = "empty/null"
        elif has_null:
            null_type = "null"
        elif has_empty:
            null_type = "empty"
        else:
            null_type = "filled"
        
        result.append({
            "column": col,
            "is_unique": is_unique,
            "distinct_count": distinct_count,
            "null_type": null_type
        })
    
    return pd.DataFrame(result)