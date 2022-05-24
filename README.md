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


# 3. EDA
1. Missing values
    1. `emp_title`: 166810 (7.3%)
       - 알지 못하는 직업인 것 같다.
       - 빈도에 대한 cumulated sum threshold 이외의 값은 other로 변환하고 **embedding** 시키면 좋을 것 같다.
       - 아예 feature를 제거하는 것과 비교해볼 필요가 있음.
       - lower, period, -, &,  제거, strip 전처리 필요
    2. `emp_length`: 146873 (6.50%)
       - 87%: `emp_title` and `emp_length` isnull
       - 1%: `bemp_length` isnull only
       - 12%: `emp_title` isnull only
       - 다른 class들과 target과의 분포가 많이 다름 
       - `emp_length`가 null인 경우, 그렇지 않은 경우에 비해,
         - `loan_status=Charged Off`의 비율이 더 높고, `Current`의 비율이 높으며, `Fully Paid`의 비율이 낮다. 즉, 좀 더 힘든 상황이라는 것
         - `annual_income`의 평균이 거의 50%
         - `home_ownership=OWN`의 비율이 2배 높다.(연소득이 더 낮은데 자가의 비율이 높다. 금수저?)
         - `verification_status=Source Verified` 비율이 50% 밖에 안 됨
         - `dti`는 22% 더 높다
       - 대소문자 시작이 섞여있고, 글자가 짤려있는 것을 보면 문자인식으로 데이터를 수집한 것 같다.
       - 새로운 class(`other`)로 만드는 게 좋아보임
    3. `title`: 23323 (1.03%)
       - `purpose`와 중복된 데이터이므로 feature 제거
    4. `zip_code`: 1 (0.00%)
       - `addr_state`의 mutual info가 가장 크니 `zip_code`는 제거
    5. `dti`: 1711 (0.08%)
       - 진짜 정보를 얻을 수 없었던 거(thin filer) (`verification_status=Not Verificed` 가 86%)
       - `loan_status`의 비율
         - `dti` is not null: 1:4:5 (`Fully Paid` 많음)
         - `dti` is null: 0.5:7.5:2 (`Fully Paid` 적음)
       - 비율이 적다고 missing row를 없애버리면 얼마 없는 `loan_status=Current` 데이터를 그냥 갖다 버리는 거
       - isnull: 전부 `application_type=Joint App`
       - `dti_missing` feature를 새로 만드는 것이 좋을 것 같다. (1: `dti` is missing value, 0: `dti` is meaningful)
         - `dti`*`dti_missing` feature도 추가하여 의미를 추가해야 한다. 
       - 0 이하의 값은 0.1로 변환
         - -1 은 무슨 의미인지 잘 모르겠으나, 2개니까 0으로 봐도 큰 문제 없을 듯
         - 0은 반올림되어 0이 된 것 같다. (0.1도 있는 것을 보면)
       - log scale로 변환 (corr 증가)
    6. `annual_inc`: 0
       - 0인 경우가 사실 missing value. (0 부근의 값의 분포가 이외의 분포와 크게 다름, 146873 (6.50%))
       - 0인 경우,
         - `verification_status=Not Verified`가 거의 3배 가깝게 높음
         - `loan_status` 역시 `Current` 비율이 2배 높고, `Fully Paid` 비율이 50%도 안 됨
         - `emp_length`가 1년도 안 되는 경우가 4배 이상 높음 (알바?)
         - `grade`도 더 낮은 경향을 보인다(`A`, `B`의 비율이 5%씩 낮음)
         - `application_type=Joint App` 비율이 100%
       - 큰 특징의 차이가 있으므로, `dti`와 마찬가지로 flag feature를 만들 필요가 있음
    7. `revol_util`: 1762 (0.08%)
       - 결측치에 비해,
         - `loan_status=Current`의 비율이 25% 정도 더 높은 것을 보면 `dti` 처럼 thin filer 일지도?
         - `verification_status`는 오히려 더 상태가 좋음 (`Not Verified`의 30% 정도 더 낮음)
         - `revol_bal`의 평균이 더 높다는 것은, 평균적으로 빌릴 수 있는 돈이 더 많다? 아무튼, 연속적이지 않고 다른 형태의 분포를 가지고 있다.
       - `revol_util=0`인 경우, `revol_bal`의 분포가 봉우리가 2개인 mixture gaussian dist.로 보인다.
       - 큰 특징의 차이가 있으므로, 결측치, 0값에 대한 flag feature를 2개 만들 필요가 있음
    8. `revol_bal`: 0
       - 0값이 엄청 많음.(10249)
       - Domain의 연속성이 있는 것 같기도하고 애매
       - 0값에 대한 flag feature 만들어 놓자
    9. `mort_acc`: 47281 (2.09%)
       - 결측치에서,
         - `grade`의 `A` 비율이 30% 정도 더 높음
         - `verification_status=Not Verified`의 비율이 33% 더 높고, `Source Verified`의 비율이 25% 더 낮음
         - `loan_status=Current`가 없음. 계좌가 없으니 그런 듯?
         - `initial_list_status=f`의 비율이 100%
         - `application_type=Individual`의 비율이 100%
       - 결측치가 target 뿐만 아니라 여러 feature에 결정적인 영향을 준다. (tree model에 적합)
       - 따라서, 결측치라는 표시를 해줄 feature를 추가하는 것이 필요.
    10. `pub_rec_bankruptcies`: 697 (0.03%)
        - 결측치에서,
          - `mort_acc`과 유사하게, 하나의 class로 선택되는 feature들이 많다.
          - `term=36 months` 전부
          - `verification_status=Not Verified` 거의 전부
          - `pub_rec=0` 거의 전부
          - `initial_list_status=f` 전부
          - `application_type=Individual` 전부
          - `mort_acc`이 전부 결측치
        - 따라서, 결측치라는 표시를 해줄 feature를 추가하는 것이 필요.


# 4. Proposed idea
1. [`proposed1`](proposed1.ipynb)
