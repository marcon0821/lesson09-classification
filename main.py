import json
import os
import pandas as pd
import plotly.express as px
import streamlit as st

# 1. Page Configuration (Title and Browser Tab Icon)
st.set_page_config(
    page_title="뇌졸중 예측 실습실", page_icon="🧠", layout="wide"
)

# Constants & Paths
DATA_URL = "https://raw.githubusercontent.com/marcon0821/data/refs/heads/main/stroke.csv"
SAVE_DIR = "내_기록"
SAVE_FILE = os.path.join(SAVE_DIR, "data_info.json")

# Ensure directory exists for saving records
os.makedirs(SAVE_DIR, exist_ok=True)


# Load Saved Notes / Metadata
def load_saved_data():
    if os.path.exists(SAVE_FILE):
        try:
            with open(SAVE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"meanings": {}, "source": ""}


def save_user_data(meanings_dict, source_text):
    data_to_save = {"meanings": meanings_dict, "source": source_text}
    with open(SAVE_FILE, "w", encoding="utf-8") as f:
        json.dump(data_to_save, f, ensure_ascii=False, indent=4)


saved_records = load_saved_data()


# 2. Data Loading & Caching
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")
    return df


df = load_data()

# 3. App Header
st.title("🧠 뇌졸중 예측 실습실")
st.markdown(
    "본 실습실에서는 뇌졸중 관련 데이터를 확인하고 분석 및 예측 모델을 구축할 수 있습니다."
)
st.divider()

# 4. Summary Metrics Cards
total_people = len(df)
total_cols = len(df.columns)
stroke_count = int(df["stroke"].sum())
stroke_ratio = (stroke_count / total_people) * 100

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(label="👥 전체 사람 수", value=f"{total_people:,} 명")

with col2:
    st.metric(label="📊 열(컬럼) 개수", value=f"{total_cols} 개")

with col3:
    st.metric(label="🚨 뇌졸중 환자 수 (stroke=1)", value=f"{stroke_count:,} 명")

with col4:
    st.metric(label="📈 뇌졸중 비율", value=f"{stroke_ratio:.2f} %")

st.divider()

# 5. Column Information & Editable Meaning Section
st.subheader("📋 데이터 열(컬럼) 명세서")
st.caption(
    "교재를 참고하여 각 열의 우리말 뜻을 입력해 주세요. 입력한 내용은 자동으로 저장됩니다."
)

columns_info = []
meanings_input = {}

# Layout for editable meanings
for col in df.columns:
    null_count = df[col].isnull().sum()
    data_type = str(df[col].dtype)

    # Load saved meaning or default to empty string
    default_meaning = saved_records.get("meanings", {}).get(col, "")

    c1, c2, c3, c4 = st.columns([2, 3, 2, 2])
    with c1:
        st.markdown(f"**`{col}`**")
    with c2:
        user_meaning = st.text_input(
            f"우리말 뜻 ({col})",
            value=default_meaning,
            key=f"meaning_{col}",
            label_visibility="collapsed",
            placeholder="우리말 뜻을 입력하세요...",
        )
        meanings_input[col] = user_meaning
    with c3:
        st.write(f"값 종류: `{data_type}`")
    with c4:
        if null_count > 0:
            st.write(f"⚠️ 결측치: `{null_count}개`")
        else:
            st.write("✅ 결측치 없음")

st.divider()

# 6. Preview Data (First 5 Rows)
st.subheader("🔍 데이터 미리보기 (상위 5개 행)")
st.dataframe(df.head(5), use_container_width=True)

st.divider()

# 7. Editable Source Section
st.subheader("📌 데이터 출처")
st.caption(
    "교재에 적힌 데이터 출처를 아래에 작성해 주세요. 작성한 내용은 저장되어 유지됩니다."
)

default_source = saved_records.get("source", "")
source_text = st.text_area(
    "데이터 출처 입력",
    value=default_source,
    placeholder="데이터 출처 내용을 입력해 주세요...",
    height=120,
    label_visibility="collapsed",
)

# 8. Save Button logic
if st.button("💾 입력한 내용 저장하기", type="primary"):
    save_user_data(meanings_input, source_text)
    st.success("우리말 뜻과 데이터 출처가 '내_기록/data_info.json'에 저장되었습니다!")
