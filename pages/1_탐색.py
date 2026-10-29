import pandas as pd
import plotly.express as px
import streamlit as st

# 1. 페이지 설정
st.set_page_config(
    page_title="데이터 탐색 - 뇌졸중 예측 실습실", page_icon="📊", layout="wide"
)

# 데이터 URL
DATA_URL = "https://raw.githubusercontent.com/marcon0821/data/refs/main/stroke.csv"
LOCAL_FILE = "stroke.csv"


# 데이터 로드 함수 (캐싱 적용)
@st.cache_data
def load_data():
    try:
        df = pd.read_csv(DATA_URL, encoding="utf-8")
        return df
    except Exception as e_url:
        try:
            df = pd.read_csv(LOCAL_FILE, encoding="utf-8")
            return df
        except Exception as e_local:
            st.error(
                f"❌ 데이터를 불러오지 못했습니다!\n\n"
                f"1. 웹 URL 오류: {e_url}\n"
                f"2. 로컬 파일 오류: {e_local}\n\n"
                f"GitHub 링크 주소를 확인하거나 `stroke.csv` 파일을 확인해주세요."
            )
            st.stop()


df = load_data()

# 2. 페이지 제목
st.title("📊 뇌졸중 데이터 탐색 페이지")
st.markdown("수집된 뇌졸중 데이터를 바탕으로 다양한 특성 간의 관계를 탐색합니다.")
st.divider()

# 3. 나이와 평균 혈당 분포 히스토그램 (나란히 배치)
st.subheader("1️⃣ 나이 및 평균 혈당 분포")
col1, col2 = st.columns(2)

with col1:
    fig_age = px.histogram(
        df,
        x="age",
        nbins=30,
        title="나이(age) 분포",
        labels={"age": "나이", "count": "사람 수"},
        color_discrete_sequence=["skyblue"],
    )
    st.plotly_chart(fig_age, use_container_width=True)

with col2:
    fig_glucose = px.histogram(
        df,
        x="avg_glucose_level",
        nbins=30,
        title="평균 혈당(avg_glucose_level) 분포",
        labels={"avg_glucose_level": "평균 혈당", "count": "사람 수"},
        color_discrete_sequence=["salmon"],
    )
    st.plotly_chart(fig_glucose, use_container_width=True)

st.divider()

# 4. 뇌졸중 여부에 따른 나이 및 혈당 상자그림(Box Plot) 및 평균값 표
st.subheader("2️⃣ 뇌졸중 여부에 따른 나이 및 평균 혈당 비교")
# 0과 1을 알기 쉬운 레이블로 변환한 임시 컬럼 생성
df["stroke_label"] = df["stroke"].map({0: "정상 (0)", 1: "뇌졸중 (1)"})

col3, col4 = st.columns(2)

with col3:
    fig_box_age = px.box(
        df,
        x="stroke_label",
        y="age",
        title="뇌졸중 유무에 따른 나이 분포",
        labels={"stroke_label": "뇌졸중 여부", "age": "나이"},
        color="stroke_label",
    )
    st.plotly_chart(fig_box_age, use_container_width=True)

with col4:
    fig_box_glucose = px.box(
        df,
        x="stroke_label",
        y="avg_glucose_level",
        title="뇌졸중 유무에 따른 평균 혈당 분포",
        labels={"stroke_label": "뇌졸중 여부", "avg_glucose_level": "평균 혈당"},
        color="stroke_label",
    )
    st.plotly_chart(fig_box_glucose, use_container_width=True)

# 두 그룹의 평균값 표 제공
st.markdown("##### 📋 그룹별 평균값 비교 표")
mean_table = (
    df.groupby("stroke_label")[["age", "avg_glucose_level"]]
    .mean()
    .reset_index()
)
mean_table.columns = ["뇌졸중 여부", "평균 나이", "평균 혈당"]
st.dataframe(
    mean_table.style.format(
        {"평균 나이": "{:.1f}세", "평균 혈당": "{:.2f}"}
    ),
    use_container_width=True,
)

st.divider()

# 5. 고혈압 및 심장병 여부에 따른 뇌졸중 비율 막대그래프
st.subheader("3️⃣ 고혈압 및 심장병과 뇌졸중 비율")


# 비율 계산용 함수
def get_stroke_rate_df(col_name, col_display_name):
    temp_df = (
        df.groupby(col_name)["stroke"]
        .agg(
            총인원="count",
            뇌졸중환자수="sum",
        )
        .reset_index()
    )
    temp_df["뇌졸중 비율 (%)"] = (
        temp_df["뇌졸중환자수"] / temp_df["총인원"]
    ) * 100
    temp_df[col_name] = temp_df[col_name].map({0: "없음 (0)", 1: "있음 (1)"})
    temp_df.rename(columns={col_name: col_display_name}, inplace=True)
    return temp_df


col5, col6 = st.columns(2)

with col5:
    hypertension_df = get_stroke_rate_df("hypertension", "고혈압 여부")
    fig_hyp = px.bar(
        hypertension_df,
        x="고혈압 여부",
        y="뇌졸중 비율 (%)",
        text="뇌졸중 비율 (%)",
        title="고혈압 여부에 따른 뇌졸중 비율",
        color="고혈압 여부",
        color_discrete_sequence=["#636EFA", "#EF553B"],
    )
    fig_hyp.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
    st.plotly_chart(fig_hyp, use_container_width=True)

with col6:
    heart_df = get_stroke_rate_df("heart_disease", "심장병 여부")
    fig_heart = px.bar(
        heart_df,
        x="심장병 여부",
        y="뇌졸중 비율 (%)",
        text="뇌졸중 비율 (%)",
        title="심장병 여부에 따른 뇌졸중 비율",
        color="심장병 여부",
        color_discrete_sequence=["#636EFA", "#EF553B"],
    )
    fig_heart.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
    st.plotly_chart(fig_heart, use_container_width=True)

st.divider()

# 6. 체질량지수(bmi) 결측치 그룹과 전체 뇌졸중 비율 비교
st.subheader("4️⃣ 체질량지수(bmi) 결측치 그룹 vs 전체 뇌졸중 비율 비교")

total_count = len(df)
total_stroke_count = df["stroke"].sum()
total_stroke_rate = (total_stroke_count / total_count) * 100

bmi_null_df = df[df["bmi"].isnull()]
null_count = len(bmi_null_df)
null_stroke_count = bmi_null_df["stroke"].sum()
null_stroke_rate = (
    (null_stroke_count / null_count) * 100 if null_count > 0 else 0
)

comparison_data = {
    "구분": ["전체 데이터", "bmi 결측치(빈 값) 데이터"],
    "전체 사람 수": [total_count, null_count],
    "뇌졸중 환자 수": [total_stroke_count, null_stroke_count],
    "뇌졸중 비율 (%)": [f"{total_stroke_rate:.2f}%", f"{null_stroke_rate:.2f}%"],
}

comparison_table = pd.DataFrame(comparison_data)
st.dataframe(comparison_table, use_container_width=True)

st.divider()

# 7. 흡연 상태(smoking_status) 값마다 사람 수를 센 표
st.subheader("5️⃣ 흡연 상태(smoking_status)별 인원 수")

smoking_counts = (
    df["smoking_status"]
    .fillna("Unknown (결측)")
    .value_counts()
    .reset_index()
)
smoking_counts.columns = ["흡연 상태", "사람 수"]

col_a, col_b = st.columns([1, 1])
with col_a:
    st.dataframe(smoking_counts, use_container_width=True)
with col_b:
    fig_smoke = px.pie(
        smoking_counts,
        names="흡연 상태",
        values="사람 수",
        title="흡연 상태 비율",
        hole=0.4,
    )
    st.plotly_chart(fig_smoke, use_container_width=True)
