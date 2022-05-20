# Lending club dataset(2007~2018)에 대한 신용평가시스템

# 1. Dataset
## 1.1 Description
1. Lending Club에서 제공한 데이터(현재는 직접 다운로드 불가, [참고](https://www.kaggle.com/datasets/wordsforthewise/lending-club/discussion/317467))를 [가공](https://github.com/nateGeorge/preprocess_lending_club_data)한 데이터
   - `int_rate`, `revol_util` columns의 단위를 `%`에서 floats로 변환
   - 데이터에 약간의 문제가 있다는 얘기도 있다([참고](https://www.kaggle.com/datasets/wordsforthewise/lending-club/discussion/170691))
2. 기간: 2007년부터 2018년 
3. 2개의 csv파일로 구성
   - 대출 허용: `accepted_2007_to_2018Q4.csv`
     - FICO score 포함
   - 대출 거절: `rejected_2007_to_2018Q4.csv`
4. Features에 대한 설명: [https://resources.lendingclub.com/LCDataDictionary.xlsx](https://resources.lendingclub.com/LCDataDictionary.xlsx)


## 1.2 Download
1. Manual download \
[All Lending Club loan data](https://www.kaggle.com/datasets/wordsforthewise/lending-club)에서 다운로드

2. [`kaggle` API]((https://www.kaggle.com/docs/api)) 사용 \
[`download_dataset.sh`](download_dataset.sh)를 실행하여 `input` directory 안에 데이터를 다운로드할 수 있음
    ```
    input
    ├── LCDataDictionary.xlsx
    ├── accepted_2007_to_2018Q4.csv
    └── rejected_2007_to_2018Q4.csv
    ```

# 2. Baseline code
[🏦 Lending Club Loan 💰 Defaulters 🏃‍♂ Prediction](https://www.kaggle.com/code/faressayah/lending-club-loan-defaulters-prediction)
