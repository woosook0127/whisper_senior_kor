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

- 모델: whisper tiny, medium, base model. 을 fine-tuning
- dataset: KsponSpeech. 1000h 한국어 비정형 음성 데이터셋
- 성능: KsponSpeech evaluation dataset에 대하여 medium model이 7.61, 8.36의 CER, 26시간 소요

## 2.2 [github](https://github.com/rtzr/Awesome-Korean-Speech-Recognition?tab=readme-ov-file#왜-cer로-계산하나요-character-error-rate)

---

# 3. 방법론

## 3-1. 모델

## 3-2. Dataset

## 3-3. 평가지표

- 한국어는 교착어(첨가어)로 조사를 사용하고 다른 언어와 비교하여 형태소의 구조가 복잡하며, 띄어쓰기가 모호한 부분이 많기 때문에 단어 수준에서의 평가가 어렵다. 따라서 WER 보다는 CER이 적합하다고 판단.
- 

---

# 4. 실험

## 4-1. 실험환경 설정

## 4-2. Whisper model의 zero-shot 평가

## 4-3. Whisper model의 fine-tuning

## 4-4. Fine-tuned model의 성능 비교

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/6b3ed91b-8046-4da5-9275-98597568e3f1/2d3a217c-346d-4b21-abd3-d944be15541a/Untitled.png)

---

# 5. 결론
