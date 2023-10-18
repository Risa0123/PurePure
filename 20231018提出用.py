import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import streamlit as st # フロントエンドを扱うstreamlitの機能をインポート

#matplotlibで日本語を記述
matplotlib.rcParams['font.family'] = 'MS Gothic'  # Windowsの場合

# CSV ファイルの読み込み ここは毎日更新するように改造
file_path = 'data_20230925_120000.csv'
df = pd.read_csv(file_path)

# フォントの指定
st.markdown(
    """
    <style>
        .css-1ov9vh9 {
            font-family: 'MS Gothic', sans-serif;
            font-size: 20px;
        }
    </style>
    """,
    unsafe_allow_html=True
)


st.title('赤羽注目物件') # タイトル
st.write("急いで物件を決めないといけない方へPurePure不動産が物件探しをお手伝いします。")
st.markdown("***")

#警告を無効にする
st.set_option('deprecation.showPyplotGlobalUse', False)

st.write("「人気」といわれたけど、本当に人気物件なの？\n\n⇒ PurePure不動産は実際の物件の動きを分析し、データに基づいた「人気条件」を提示します。")

# 説明文を表示
st.markdown("***")
st.write(f"■ PurePureは赤羽駅から徒歩10分圏内・ワンルーム・1K・1DK・1LDKの物件の中から\n\n人気条件を絞り込んでいます。\n\n■ 同条件の物件平均価格やPurePureのデータベースから分析した\n\n独自の「人気物件」をご紹介いたします。")
st.markdown("***")

# Streamlit アプリ
st.title("あなたの優先順位は？")

# ユーザーが関数を選択できるようにドロップダウンメニューを作成
option = st.radio('優先順位が高いものを一つ選択してください。', ("広さ優先", "安さ優先", "近さ優先"))

st.text(option)

st.markdown("***")
st.title("あなたへおすすめの物件を最大5件を表示します。")

#アルゴリズム
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier

# Load the data from the uploaded CSV file
file_path = 'data_20230925_120000_vs_data_20230926_120000_dif_20231009_141527_dif_20231009_154237.csv'
data = pd.read_csv(file_path)

# Calculate the 'total_fee' by dividing 'management_fee' by 10000 and adding it to 'fee'
data['total_fee'] = data['fee'] + (data['management_fee'] / 10000)

# Reorder the columns to place 'total_fee' to the left of 'sold'
column_order = ['floor', 'fee', 'management_fee', 'total_fee', 'deposit', 'gratuity', 'madori', 'menseki', 
                'years_built', 'floors', 'walking_time_from_jr_akabane', 'sold']
data = data[column_order]

filename ="data_20230925.csv"
data.to_csv(filename, index=False)

#欠損値評価
# Check for missing values in the dataset
missing_values = data.isnull().sum()

# Basic statistics of the dataset
basic_stats = data.describe()

# Distribution of the target variable 'sold'
target_distribution = data['sold'].value_counts(normalize=True)

# Separate features and target variable from the dataset
X = data.drop(columns=['sold'])
y = data['sold']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the RandomForestClassifier
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)

# Get feature importances
feature_importances = rf_clf.feature_importances_

# Create a DataFrame for feature importances
feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

#GradientBoostingClassifierモデル

# Select features with importance greater than 0.1
selected_features = feature_importance_df[feature_importance_df['Importance'] > 0.1]['Feature'].tolist()

# Create new training and testing sets using only selected features
X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]

# Initialize RandomForest and GradientBoosting classifiers
rf_clf_selected = RandomForestClassifier(n_estimators=100, random_state=42)
gb_clf = GradientBoostingClassifier(n_estimators=100, random_state=42)

# Train the classifiers
rf_clf_selected.fit(X_train_selected, y_train)
gb_clf.fit(X_train_selected, y_train)

# Make predictions
rf_predictions = rf_clf_selected.predict(X_test_selected)
gb_predictions = gb_clf.predict(X_test_selected)

# Evaluate the classifiers
rf_accuracy = accuracy_score(y_test, rf_predictions)
gb_accuracy = accuracy_score(y_test, gb_predictions)

# Add the new feature to the selected features list
selected_features_with_new = selected_features + ['total_fee']

# Create new training and testing sets using selected features and the new feature
X_train_selected_with_new = X_train[selected_features_with_new]
X_test_selected_with_new = X_test[selected_features_with_new]

# Initialize RandomForest and GradientBoosting classifiers
rf_clf_with_new = RandomForestClassifier(n_estimators=100, random_state=42)
gb_clf_with_new = GradientBoostingClassifier(n_estimators=100, random_state=42)

# Train the classifiers
rf_clf_with_new.fit(X_train_selected_with_new, y_train)
gb_clf_with_new.fit(X_train_selected_with_new, y_train)

# Make predictions
rf_predictions_with_new = rf_clf_with_new.predict(X_test_selected_with_new)
gb_predictions_with_new = gb_clf_with_new.predict(X_test_selected_with_new)

# Evaluate the classifiers
rf_accuracy_with_new = accuracy_score(y_test, rf_predictions_with_new)
gb_accuracy_with_new = accuracy_score(y_test, gb_predictions_with_new)

#　GBを使った予測モデル
# Make predictions for the entire test set using Gradient Boosting model
gb_predictions_all_test = gb_clf_with_new.predict(X_test_selected_with_new)

# Create a DataFrame to store the test data along with predictions
test_data_with_predictions = X_test.copy()
test_data_with_predictions['GB_Predictions'] = gb_predictions_all_test

# Sort the DataFrame based on 'menseki' and 'GB_Predictions'
sorted_recommendations = test_data_with_predictions.sort_values(by=['GB_Predictions', 'menseki'], ascending=[False, False])

# Show the top 5 recommended properties that are likely to be sold and are spacious
top_5_recommended_properties = sorted_recommendations.head(5)
#top_5_recommended_properties

# Filter the DataFrame to include only properties that are likely to be sold (GB_Predictions = 1)
recommended_sold_properties = test_data_with_predictions[test_data_with_predictions['GB_Predictions'] == 1]

# 部屋の面積優先
def priority_menseki():
# Filter the DataFrame to include only the properties likely to be sold (GB_Predictions = 1)
    recommended_sold_properties = test_data_with_predictions[test_data_with_predictions['GB_Predictions'] == 1]

# Sort these properties by 'menseki' to find the spacious ones
    sorted_recommended_sold_properties = recommended_sold_properties.sort_values(by='menseki', ascending=False)

# Show the top 5 recommended properties that are likely to be sold (GB_Predictions = 1) and are spacious
    top_5_recommended_sold_properties = sorted_recommended_sold_properties.head(5)
    return top_5_recommended_sold_properties

#安くて比較的駅から近い
def priority_cheapandclose():
# Filter the DataFrame to include only properties that are likely to be sold (GB_Predictions = 1)
    recommended_sold_properties = test_data_with_predictions[test_data_with_predictions['GB_Predictions'] == 1]

# Sort these properties by 'fee' and 'walking_time_from_jr_akabane' to find the affordable and near-station ones
    sorted_affordable_near_station_properties = recommended_sold_properties.sort_values(by=['fee', 'walking_time_from_jr_akabane'])

# Show the top 5 recommended properties that are likely to be sold, affordable, and near the station
    top_5_affordable_near_station_properties = sorted_affordable_near_station_properties.head(5)
    return top_5_affordable_near_station_properties

# Filter the DataFrame to include only properties that are likely to be sold (GB_Predictions = 1)

#駅から１０分以内
def priority_10min():
# and are within 10 minutes from the station
    recommended_sold_near_station_properties = recommended_sold_properties[recommended_sold_properties['walking_time_from_jr_akabane'] <= 10]

# Sort these properties by 'fee' to find the affordable ones
    sorted_affordable_very_near_station_properties = recommended_sold_near_station_properties.sort_values(by='fee')

# Show the top 5 recommended properties that are likely to be sold, affordable, and very near the station (within 10 minutes)
    top_5_affordable_very_near_station_properties = sorted_affordable_very_near_station_properties.head(5)
    return top_5_affordable_very_near_station_properties


# 選択に応じて関数を実行
if option == "広さ優先":
    top_properties = priority_menseki()
    st.write("Top 5 部屋の広さ優先物件:")
    st.write(top_properties)

elif option == "安さ優先":
    top_properties =priority_cheapandclose()
    st.write("Top 5 部屋の安さ優先物件:")
    st.write(top_properties)


elif option == "近さ優先":
    top_properties =priority_10min()
    st.write("Top 5 部屋の近さ優先物件:")
    st.write(top_properties)


if st.button('気になる物件番号を問い合わせる'):
    property_number = st.text_input("気になる物件の番号を入力してください:")
    
    # ユーザーが入力した物件番号を使って何か処理を行うことができます
    # 例: データベースから該当する情報を取得して表示する
    
    if property_number:
        st.write(f"入力された物件番号: {property_number}")

st.markdown("***")

# 分析とグラフ作成
#フォント指定
plt.rcParams['font.sans-serif'] = ['MS Gothic']
plt.clf()

# Scatter Plot 1: Walking Time from JR Akabane vs Rent Fee
def scatter_walking_time_fee():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(df['walking_time_from_jr_akabane'], df['fee'], c='b', alpha=0.6, label='Properties')
    ax.set_xlabel('Walking Time from JR Akabane (min)')
    ax.set_ylabel('Rent Fee (x10,000 JPY)')
    ax.set_title('Relationship between Walking Time from JR Akabane and Rent Fee')
    ax.set_xlabel('赤羽駅からの徒歩時間(分)')  # X軸のラベルを変更
    ax.set_ylabel('賃貸料（円)')  # Y軸のラベルを変更
    ax.set_title(u'賃貸料と駅からの時間の関係')  # タイトルを変更
    ax.set_ylim([0, 40])
    st.pyplot(fig)
    plt.clf()  # グラフの状態をクリア

# Scatter Plot 2: Walking Time from JR Akabane vs Area (m^2)
def scatter_walking_time_menseki():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(df['walking_time_from_jr_akabane'], df['menseki'], c='g', alpha=0.6, label='Properties')
    ax.set_xlabel('Walking Time from JR Akabane (min)')
    ax.set_ylabel('Area (m^2)')
    ax.set_title('Relationship between Walking Time from JR Akabane and Area')
    ax.set_xlabel('赤羽駅からの徒歩時間(分)')  # X軸のラベルを変更
    ax.set_ylabel('部屋面積（m2)')  # Y軸のラベルを変更
    ax.set_title(u'部屋の広さと駅からの時間の関係')  # タイトルを変更
    ax.set_ylim([0, 300])
    st.pyplot(fig)
    plt.clf()  # グラフの状態をクリア

# Scatter Plot 3: Years Built vs Rent Fee
def scatter_years_built_fee():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(df['years_built'], df['fee'], c='r', alpha=0.6, label='Properties')
    ax.set_xlabel('Years Built')
    ax.set_ylabel('Rent Fee (x10,000 JPY)')
    ax.set_title('Relationship between Years Built and Rent Fee')
    ax.set_xlabel('築年数')  # X軸のラベルを変更
    ax.set_ylabel('賃貸料(円)')  # Y軸のラベルを変更
    ax.set_title(u'築年数と賃貸料の関係')  # タイトルを変更
      # フォントサイズを指定
    fontsize = 16

    ax.set_ylim([0, 40])
    st.pyplot(fig)
    plt.clf()  # グラフの状態をクリア

# Boxplot 1: Layout (Madori) vs Walking Time from JR Akabane
def boxplot_layout_walking_time():
    fig, ax = plt.subplots(figsize=(15, 8))
    sns.boxplot(x='madori', y='walking_time_from_jr_akabane', data=df, palette='Set3')
    ax.set_xlabel('Layout (Madori)')
    ax.set_ylabel('Walking Time from JR Akabane (min)')
    ax.set_title('Relationship between Layout and Walking Time from JR Akabane')
    ax.set_xlabel('間取り')  # X軸のラベルを変更
    ax.set_ylabel('赤羽駅からの徒歩時間(分)')  # Y軸のラベルを変更
    ax.set_title(u'部屋の間取りと駅からの時間の関係')  # タイトルを変更
          # フォントサイズを指定
    fontsize = 16
    st.pyplot(fig)
    plt.clf()  # グラフの状態をクリア

# Boxplot 2: Layout (Madori) vs Rent Fee
def boxplot_layout_fee():
    fig, ax = plt.subplots(figsize=(15, 8))
    sns.boxplot(x='madori', y='fee', data=df, palette='Set2')
    ax.set_xlabel('Layout (Madori)')
    ax.set_ylabel('Rent Fee (x10,000 JPY)')
    ax.set_title('Relationship between Layout and Rent Fee')
    ax.set_xlabel('間取り')  # X軸のラベルを変更
    ax.set_ylabel('賃貸料(円)')  # Y軸のラベルを変更
    ax.set_title(u'部屋の間取りと賃貸料の関係')  # タイトルを変更
      # フォントサイズを指定
    fontsize = 16
    ax.set_ylim([0, 50])
    st.pyplot(fig)
    plt.clf()  # グラフの状態をクリア

# Scatter Plot 4: Area (m^2) vs Rent Fee
def scatter_menseki_fee():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(df['menseki'], df['fee'], c='purple', alpha=0.6, label='Properties')
    ax.set_xlabel('Area (m^2)')
    ax.set_ylabel('Rent Fee (x10,000 JPY)')
    ax.set_title('Relationship between Area and Rent Fee')
    ax.set_xlabel('部屋面積（m2)')  # X軸のラベルを変更
    ax.set_ylabel('賃貸料(円)')  # Y軸のラベルを変更
    ax.set_title(u'部屋の広さと賃貸料の関係')  # タイトルを変更
          # フォントサイズを指定
    
    fontsize = 16
    ax.set_ylim([0, 40])
    st.pyplot(fig)
    plt.clf()  # グラフの状態をクリア

st.title('物件選びの豆知識')
st.write("安い物件は、狭くて、古くて、駅から遠くてもしょうがないと思っていませんか？\n\nそんなことはありません! PurePureはデータからあなたに合わせた物件をお選びします！")
st.markdown("***")

# グラフをStreamlitへ反映追加するために追加
def main():
    st.header('賃貸料と駅からの時間の関係')
    st.write("駅に近いほど家賃が高い印象がありましたが、必ずしもそうではないことが分かります。")
    scatter_walking_time_fee()
        
    st.header('部屋の広さと駅からの時間の関係')
    st.write("駅近物件は、狭い部屋が多い印象ですが、そうではありまん。")
    scatter_walking_time_menseki()

    st.header('賃貸料と築年数の関係')
    st.write("築年数が古いと賃料が安い印象ですが、そうではないようです。\n\n 築年数が古くてもリノベーション物件などで、きれいで住みやすい物件もあります。")
    scatter_years_built_fee()

    st.header('間取りと駅からの時間の関係')
    st.write("1LDKの物件が一番ばらつきが多い結果になりました。")
    boxplot_layout_walking_time()

    st.header('間取りと賃貸料の関係')
    st.write("1K、ワンルーム、2K、1DKは大きな差がなさそうな結果が出ました。")
    boxplot_layout_fee()

    st.header('部屋の面積と賃貸料の関係')
    st.write("ある一定までは面積と賃貸料が比例していますが、途中から異なってきていますね。")
    scatter_menseki_fee()

if __name__ == '__main__':
    main()
