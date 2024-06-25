
# 1. 서론

- 자동음성인식(Automatic Speech Recognition, ASR or Speech To Text, STT)
- 최근 음성인식을 위한 딥러닝 모델은 Transformer 모델을 기본으로 발전하고있다.
    - 모델의 구조가 복잡해지고 크기가 커짐에 따라 대규모 데이터를 이용하여 사전 학습된 LLM 을 이용하여 각 응용 분야에 맞는 모델을 fine-tuning하는 학습 방법이 일반화되었다.
    - LLM은 대규모의 non-labeled data에 대해 자기지도학습(Self-Supervised Learning, SSL)을 수행하여 생성된다.
- Whisper model
    - OpenAI에서 공개한 LLM.
        - 대규모 데이터를 자동으로 정제하여 대규모의 Transcription data를 생성하여 학습하는 방식을 적용
        - Fine- tuning 없이도 훌륭한 ASR 성능을 보인다.
            - Noisy 환경이나, 비정형 자유발화 등 기존 ASR model이  어려워하는 영역에서 잘 동작한다
            - 다국어 음성인식을 지원한다.
        - 단점
            - 다국어 음성인식 모델의 고질적 문제인 데이터의 양질의 문제가 robust한 다국어 ASR의 성능을 저해한다.
            - 언어마다 인식률이 다르며, 일부 언어에서는 WER 이 30% 를 초과한다.
- 한국어 음성인식 중에서도 노인 및 방언 음성 인식을 fine-tuned whisper model 을 baseline model로 삼아 이를 fine-tuning하여 성능 향상을 목표로 한다.

---

# 2. 관련 연구

## 2.1 **대형 사전훈련 모델의 파인튜닝을 통한 강건한 한국어 음성인식 모델 구축**

- 목적
      - Whisper model의 fine-tuning과 사전 학습 모델 없이 full-training한 Transformer model 성능 비교 

- Baseline model
    - Whisper tiny, Whisper medium, Whisper base

- Used dataset
    - KsponSpeech: 1000시간 한국어 비정형 음성 데이터셋

- 성능평가
    - Medium model을 KsponSpeech evaluation dataset에 대하여 측정한 CER
        - kspon-evalclean: 7.61%
        - kspon-evalother: 8.36%

---

# 3. 방법론
## 3.1. Baseline Model
    - Whisper-small_korean-zeroth
        - openai/whisper-small model’s fine-tuned version
        - WER: 19.9 %

## 3.2. Datasets
- 자유대화 음성(노인남여)
    - 60세 이상의 남녀를 대상으로 3000여 시간의 음성 데이터를 utterance 단위로 수집
    - 1000 명 이상의 발화자
    - 남녀 비율 1:1
- Train dataset
    1.Training/raw_data/1.AI 챗봇/1.AI챗봇_2_자유대화(노인남여)_TRAINING
    위 디렉토리 내의 수도권, 경상, 강원, 충청, 전라, 기타 지역 데이터에서 각각 남, 여 음성을 선택해 training input으로 사용
    # of Samples: 10715

- Validation dataset
    Train dataset에서 train_test_split(test_size=0.2)를 사용하여 분할하여 validation set으로 사용
    # of Samples: 2679

- Test dataset
    2.Validation/raw_data/1.AI챗봇/1.AI챗봇_라벨링_자유대화(노인남여)_VALIDATION
    위 디렉토리 내의 수도권, 경상, 강원, 충청, 전라, 기타 지역 데이터에서 각각 남, 여 음성을 test input으로 사용
    # of Samples: 9944

## 3.3. 평가지표
- CER (Character Error Rate)
    - 인식된 문장과 실제 정답 문장 사이의 차이를 나타내는 비율
    - 문자 단위로 오류 측정
    - S: 치환 오류의 수 , D: 삭제 오류의 수 , I:  삽입 오류의 수 , N: 정답 단어의 수

- 선정 이유
    - 한국어는 교착어(첨가어)로 조사를 사용하고, 다른 언어와 비교하여 형태소의 구조가 복잡하다
    - 띄어쓰기가 모호한 부분이 많다. 따라서 WER 보다는 CER이 더 적합하다고 판단

---

# 4. 실험
## 4.1. Baseline Model의 zero-shot 평가
- Test dataset	
    - 2.Validation/raw_data/1.AI챗봇/1.AI챗봇_라벨링_자유대화노인남여)_VALIDATION

- CER
    - 30.33 %
    - 
## 4.3. Whisper model의 fine-tuning

## 4.4. Fine-tuned model의 성능 비교

---

# 5. 결론
