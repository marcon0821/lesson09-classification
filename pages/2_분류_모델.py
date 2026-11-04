import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_graphviz

# 1. 페이지 설정
st.set_page_config(
    page_title="뇌졸중 예측 모델 실습실", page_icon="🤖", layout="wide"
)

# 데이터 URL
DATA_URL = "https://raw.githubusercontent.com/marcon0821/data/refs/heads/main/stroke.csv"
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
st.title("🤖 뇌졸중 예측 분류 모델 실습실")
st.markdown(
    "머신러닝(로지스틱 회귀 및 의사결정트리)을 활용하여 뇌졸중 여부를 예측하고 모델의 성능을 비교합니다."
)
st.divider()

# 속성 매핑 사전 (영문 열 이름 <-> 우리말 이름)
feature_mapping = {
    "age": "나이",
    "avg_glucose_level": "평균 혈당",
    "bmi": "체질량지수",
    "hypertension": "고혈압",
    "heart_disease": "심장병",
}

reverse_feature_mapping = {v: k for k, v in feature_mapping.items()}
all_korean_features = list(feature_mapping.values())
default_korean_features = [
    feature_mapping[f] for f in ["age", "avg_glucose_level", "hypertension", "heart_disease"]
]

# 3. 입력 속성 선택 UI
st.subheader("⚙️ 1. 모델 입력 속성 선택")
selected_korean_features = st.multiselect(
    "학습 및 예측에 사용할 속성을 선택해 주세요 (2개 이상)",
    options=all_korean_features,
    default=default_korean_features,
)

# 2개보다 적게 고르면 안내문 표시 후 멈춤
if len(selected_korean_features) < 2:
    st.warning("⚠️ 모델을 학습하려면 **속성을 2개 이상** 선택해 주세요.")
    st.stop()

# 선택된 영문 열 이름 리스트
selected_features = [reverse_feature_mapping[k] for k in selected_korean_features]

# 4. 데이터 전처리 (결측치 처리 및 분할 규칙 적용)
df_sorted = df.sort_values(by="id").reset_index(drop=True)

test_indices = []
train_indices = []

for i in range(0, len(df_sorted), 10):
    chunk = df_sorted.iloc[i : i + 10]
    if len(chunk) >= 3:
        test_indices.extend(chunk.index[:3])
        train_indices.extend(chunk.index[3:])
    else:
        test_indices.extend(chunk.index[:])

df_train = df_sorted.loc[train_indices].copy()
df_test = df_sorted.loc[test_indices].copy()

# bmi 결측치 처리: 훈련 데이터의 bmi 중앙값으로 채우기
if "bmi" in selected_features:
    bmi_median = df_train["bmi"].median()
    df_train["bmi"] = df_train["bmi"].fillna(bmi_median)
    df_test["bmi"] = df_test["bmi"].fillna(bmi_median)

X_train = df_train[selected_features]
y_train = df_train["stroke"]

X_test = df_test[selected_features]
y_test = df_test["stroke"]

# 스케일링 (로지스틱 회귀용)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. 모델 학습
# 1) 로지스틱 회귀 모델
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train_scaled, y_train)

log_train_acc = accuracy_score(y_train, log_reg.predict(X_train_scaled))
log_test_acc = accuracy_score(y_test, log_reg.predict(X_test_scaled))

# 2) 의사결정트리 모델
dt_clf = DecisionTreeClassifier(
    max_depth=3, min_samples_leaf=5, random_state=42
)
dt_clf.fit(X_train, y_train)

dt_train_acc = accuracy_score(y_train, dt_clf.predict(X_train))
dt_test_acc = accuracy_score(y_test, dt_clf.predict(X_test))

# 3) 베이스라인 모델 (다수 클래스 기준)
majority_class = y_train.mode()[0]
dummy_train_preds = np.full_like(y_train, majority_class)
dummy_test_preds = np.full_like(y_test, majority_class)

dummy_train_acc = accuracy_score(y_train, dummy_train_preds)
dummy_test_acc = accuracy_score(y_test, dummy_test_preds)

# 6. 정확도 결과 큰 숫자 카드 표시
st.divider()
st.subheader("📊 2. 모델 성능 비교 (정확도)")

m_col1, m_col2, m_col3 = st.columns(3)

with m_col1:
    st.metric(
        label="로지스틱 회귀 (확률로 답하는 모델)",
        value=f"{log_test_acc * 100:.2f}%",
        delta=f"훈련 정확도: {log_train_acc * 100:.2f}%",
    )

with m_col2:
    st.metric(
        label="의사결정트리 (질문으로 답하는 모델)",
        value=f"{dt_test_acc * 100:.2f}%",
        delta=f"훈련 정확도: {dt_train_acc * 100:.2f}%",
    )

with m_col3:
    st.metric(
        label="다수 클래스 기준 모델 (입력 미확인)",
        value=f"{dummy_test_acc * 100:.2f}%",
        delta=f"훈련 정확도: {dummy_train_acc * 100:.2f}%",
    )

st.divider()

# 7. 산점도 및 결정 경계 시각화
st.subheader("📈 3. 2차원 시각화 및 결정 경계 (로지스틱 회귀)")
st.markdown(
    "선택한 속성 중 **가로축과 세로축으로 삼을 두 속성**을 골라 테스트 데이터의 분포와 확률 모델의 결정 경계선(0.5)을 확인합니다."
)

c_feat1, c_feat2 = st.columns(2)
with c_feat1:
    plot_x_korean = st.selectbox(
        "가로축 속성 선택",
        options=selected_korean_features,
        index=0,
    )
with c_feat2:
    default_y_idx = 1 if len(selected_korean_features) > 1 else 0
    plot_y_korean = st.selectbox(
        "세로축 속성 선택",
        options=selected_korean_features,
        index=default_y_idx,
    )

plot_x_feat = reverse_feature_mapping[plot_x_korean]
plot_y_feat = reverse_feature_mapping[plot_y_korean]

if plot_x_feat == plot_y_feat:
    st.warning("⚠️ 가로축과 세로축에 서로 다른 속성을 선택해 주세요.")
else:
    other_features = [f for f in selected_features if f not in [plot_x_feat, plot_y_feat]]
    fixed_values = {}
    fixed_desc_list = []

    for f in other_features:
        med_val = df_train[f].median()
        fixed_values[f] = med_val
        fixed_desc_list.append(f"**{feature_mapping[f]}**: 중앙값 ({med_val:.2f})")

    if fixed_desc_list:
        st.info(
            "📌 **고정된 다른 속성 값 (중앙값 기준 계산):**\n"
            + " | ".join(fixed_desc_list)
        )

    df_test_plot = df_test.copy()
    df_test_plot["실제_뇌졸중"] = df_test_plot["stroke"].map({0: "정상 (0)", 1: "뇌졸중 (1)"})

    fig = px.scatter(
        df_test_plot,
        x=plot_x_feat,
        y=plot_y_feat,
        color="실제_뇌졸중",
        labels={plot_x_feat: plot_x_korean, plot_y_feat: plot_y_korean},
        title=f"{plot_x_korean} vs {plot_y_korean} 테스트 데이터 분포 및 로지스틱 회귀 경계선",
        color_discrete_map={"정상 (0)": "#636EFA", "뇌졸중 (1)": "#EF553B"},
    )

    x_min, x_max = df_test[plot_x_feat].min(), df_test[plot_x_feat].max()
    y_min, y_max = df_test[plot_y_feat].min(), df_test[plot_y_feat].max()
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100)
    )

    grid_data = pd.DataFrame(index=np.arange(xx.ravel().shape[0]))
    for f in selected_features:
        if f == plot_x_feat:
            grid_data[f] = xx.ravel()
        elif f == plot_y_feat:
            grid_data[f] = yy.ravel()
        else:
            grid_data[f] = fixed_values[f]

    grid_scaled = scaler.transform(grid_data)
    probs = log_reg.predict_proba(grid_scaled)[:, 1].reshape(xx.shape)

    # 결정 경계선 (0.5) 그리기 (showlines 속성 사용 수정 완료)
    fig.add_trace(
        go.Contour(
            z=probs,
            x=np.linspace(x_min, x_max, 100),
            y=np.linspace(y_min, y_max, 100),
            showscale=False,
            contours=dict(coloring="lines", showlines=True, start=0.5, end=0.5, size=0.1),
            line=dict(color="black", width=2, dash="dash"),
            name="결정 경계선 (0.5)",
        )
    )

    st.plotly_chart(fig, use_container_width=True)

    if (xx.min() > x_max or xx.max() < x_min) or (yy.min() > y_max or yy.max() < y_min):
        st.markdown(
            "ℹ️ *참고: 계산된 결정 경계선이 현재 테스트 데이터의 표시 범위를 벗어나 있을 수 있습니다.*"
        )

st.divider()

# 8. 의사결정트리 구조 시각화 (Graphviz DOT)
st.subheader("🌳 4. 의사결정트리 질문 구조 시각화")
st.markdown(
    "의사결정트리 모델이 데이터를 분류하기 위해 던진 질문과 마디별 샘플 분포를 확인합니다."
)

korean_feature_names = [feature_mapping[f] for f in selected_features]

dot_data = export_graphviz(
    dt_clf,
    out_file=None,
    feature_names=korean_feature_names,
    class_names=["정상", "뇌졸중"],
    filled=True,
    rounded=True,
    special_characters=True,
    precision=2,
)

st.graphviz_chart(dot_data)

# 9. 의사결정트리 분석 요약 정보
used_features_indices = dt_clf.tree_.feature
used_features_names = [
    korean_feature_names[i] for i in used_features_indices if i != -2
]
unique_used_features = list(set(used_features_names))

is_leaf = dt_clf.tree_.children_left == -1
leaf_values = dt_clf.tree_.value[is_leaf]
total_leaves = int(np.sum(is_leaf))
leaf_classes = np.argmax(leaf_values.squeeze(axis=1), axis=1)
not_stroke_leaves = int(np.sum(leaf_classes == 0))

st.markdown("##### 📝 의사결정트리 분석 요약")
col_summary1, col_summary2 = st.columns(2)

with col_summary1:
    st.markdown(
        f"- **답을 내는 마디 총 개수:** 총 {total_leaves}개\n"
        f"- **'아님(정상)'이라고 답하는 마디 개수:** {not_stroke_leaves}개"
    )

with col_summary2:
    features_str = (
        ", ".join(unique_used_features)
        if unique_used_features
        else "없음 (단일 마디)"
    )
    st.markdown(f"- **이 나무가 실제로 물어본 속성:** {features_str}")
