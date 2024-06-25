# 1. 서론

### 1.1. 자동음성인식 (Automatic Speech Recognition, ASR or Speech To Text, STT)

- Transformer model
    - 최근 음성인식을 위한 딥러닝 모델은 Transformer 모델을 기본으로 발전하고있다.
- Large Language model (LLM)
    - 모델의 구조가 복잡해지고 크기가 커짐에 따라 대규모 데이터를 이용하여 사전 학습된 LLM 을 이용하여 각 응용 분야에 맞는 모델을 fine-tuning하는 학습 방법이 일반화되었다.
    - LLM은 대규모의 non-labeled data에 대해 자기지도학습(Self-Supervised Learning, SSL)을 수행하여 생성된다.

### 1.2. Whisper model

- OpenAI에서 공개한 LLM.
    - 대규모 데이터를 자동으로 정제하여 대규모의 Transcription data를 생성하여 학습하는 방식을 적용
- 장점
    - Fine- tuning 없이도 훌륭한 ASR 성능을 보인다.
        - Noisy 환경이나, 비정형 자유발화 등 기존 ASR model이  어려워하는 영역에서 잘 동작한다
        - 다국어 음성인식을 지원한다.
- 단점
    - 다국어 음성인식 모델의 고질적 문제인 데이터의 양질의 문제가 robust한 다국어 ASR의 성능을 저해한다.
    - 언어마다 인식률이 다르며, 일부 언어에서는 WER 이 30% 를 초과한다
- Architecture

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/6b3ed91b-8046-4da5-9275-98597568e3f1/39295a7f-e9d8-4b90-853f-980999883e16/Untitled.png)

- Encoding, Decoding block을 쌓은 전형적인 Transformer architecture를 가진다.
    - Input audio를 30초 단위로 자르고 positional encoding 하여 각 단어의 임베딩 벡터에 위치 정보를 나타내는 벡터를 더해 Encoder, Decoder의 input으로 사용한다
    - Encoder block은 multi-head attention layer와 feed forward layer로 구성되어 있고, 문장의 모든 단어를 attention한다.
        - Multi-head attention layer는 QKV벡터를 선형변환하고 적절한 계산을 거쳐 attention score를 계산한다.
        - Feed forward layer는 입력 벡터의 차원을 확장하고, 활성화 함수를 거쳐 다시 원래 차원으로 압축하고 분산된 정보를 합친다.
    - Decoder block는 자기 회귀적 방식으로 작동하며 이전 단계에서 생성된 단어를 입력으로 받아 다음 생성할 단어를 예측하는 데 사용한다.

### 1.3. 연구의 목적

- Whisper model의 한국어 음성 인식 성능 개선
- 노인 음성 인식의 성능을 개선
    - 표준어
    - 방언: 경상, 강원, 충청, 전라, 기타
        - In training: 수도권: 3189, 경상: 2869, 전라: 2008, 충청: 1782, 기타: 1594, 강원: 1056
    

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

### 3.1. Baseline Model

- Whisper-small_korean-zeroth
    - openai/whisper-small model’s fine-tuned version
    - WER: 19.9 %

### 3.2. Datasets

- 자유대화 음성(노인남여)
    - 60세 이상의 남녀를 대상으로 3000여 시간의 음성 데이터를 utterance 단위로 수집
    - 1000 명 이상의 발화자
    - 남녀 비율 1:1
    
    ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/6b3ed91b-8046-4da5-9275-98597568e3f1/58697a24-6f28-4c25-b01b-2c0e4d1f6c2f/Untitled.png)
    
- Train dataset
    - 1.Training/raw_data/1.AI 챗봇/1.AI챗봇_2_자유대화(노인남여)_TRAINING
    - 위 디렉토리 내의 수도권, 경상, 강원, 충청, 전라, 기타 지역 데이터에서 각각 남, 여 음성을 선택해 training input으로 사용
    - num of Samples: 10715
- Validation dataset
    - Train dataset에서 train_test_split(test_size=0.2)를 사용하여 분할하여 validation set으로 사용
    - num of Samples: 2679
- Test dataset
    - 2.Validation/raw_data/1.AI챗봇/1.AI챗봇_라벨링_자유대화(노인남여)_VALIDATION
    - 위 디렉토리 내의 수도권, 경상, 강원, 충청, 전라, 기타 지역 데이터에서 각각 남, 여 음성을 test input으로 사용
    - num of Samples: 9944

### 3.3. 평가지표

- CER (Character Error Rate)
    - 인식된 문장과 실제 정답 문장 사이의 차이를 나타내는 비율
    - 문자 단위로 오류 측정
    
    ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/6b3ed91b-8046-4da5-9275-98597568e3f1/a1578729-4e79-4343-b103-9f6f92edb5c8/Untitled.png)
    
    - S: 치환 오류의 수 , D: 삭제 오류의 수 , I:  삽입 오류의 수 , N: 정답 단어의 수
- 선정 이유
    - 한국어는 교착어(첨가어)로 조사를 사용하고, 다른 언어와 비교하여 형태소의 구조가 복잡하다
    - 띄어쓰기가 모호한 부분이 많다. 따라서 WER 보다는 CER이 더 적합하다고 판단

---

# 4. 실험

### 4.1. Baseline Model의 zero-shot 평가

- Test dataset
    - 2.Validation/raw_data/1.AI챗봇/1.AI챗봇_라벨링_자유대화노인남여)_VALIDATION
- CER
    - 30.33 %
        
        ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/6b3ed91b-8046-4da5-9275-98597568e3f1/69ac15a0-7986-4475-9255-41a8609bef0d/Untitled.png)
        

### 4.2. Whisper model fine-tuning

- Set Tokenizer, Feature extractor, Processor
    - Checkpoint를 사용해 사용하려는 모델에 맞는 Tokenizer, Feature extractor, Processor를 정의한다.
- Dataset preprocessing
    - WhisperFeatureExtractor를 사용해 audio array를 log-mel spectogram으로 변환한다.
    - WhisperTokenizer 를 사용해 label을 tokenize한다.
- Set Data Collator
    - DatasetDict 형식으로 load된 training data를 model의 input으로 만들기 위해 torch.tensor로 변환하고, 지정된 크기의 batch 로 만드는 class
    - raw data와 label data의 길이가 다르기 때문에 서로 다른 padding 방법을 적용한다.
        - raw data는 feature_extractor를 사용해 padding
        - label data는 tokenize된 label sequence에 대해 tokenizer를 사용해 최대 길이만큼 padding
        - padding token을 -100으로 치환하여 계산 과정에서 무시되도록 한다
        - bos token을 잘라낸다
- Set metric
    - 평가 지표로 CER을 사용하기 위해 함수를 정의
        - predict의 label_ids에서 -100을 pad_token_id로 변환
        - predict ID와 label ID를 문자열로 decoding
        - predict label과 reference label간의 CER을 계산
    - Evaluation하는 동안 발생하는 CUDA out of memory error를 막기 위해 logits의 차원 축소 필요
- Set Training arguments
    - Output directory, steps, batch size 등의 hyperparameter를 설정한다.
- Set Trainer
    - Custom dataset을 이용해 fine-tuning할 수 있도록 도움을 준다.
    - 앞서 선언한 pre-processor, model, metric, dataset을 argument로 Trainer생성

### 4.3. Training result

- CER of validation
    
    ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/6b3ed91b-8046-4da5-9275-98597568e3f1/a42e4eed-3c20-48ec-b1e3-143ec3ca0f61/Untitled.png)
    

- Training loss

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/6b3ed91b-8046-4da5-9275-98597568e3f1/0bab2899-eed1-4564-b63f-5c1a65c0f6fa/Untitled.png)

### 4.4. Trouble shooting

- 데이터셋 전처리 문제
    - 문제상황
        - Training 데이터셋의 sample 수가 약 100만개로 너무 많아서 label을 tokenization 하고 raw data를 log-mel spectogram으로 변환하여 model의 입력으로 만드는 전처리 과정에서 약 400 시간이 소요되는 문제 발생
    - 해결방법
        - 약 1만개의 training 데이터만 사용하여 전처리 속도를 약 5 분으로 줄이고, 성능 개선 또한 이뤄 냄
- 훈련 과정에서 evaluation 할 때 CUDA Out Of Memory 발생
    - 문제 상황
        - Evaluation 할 때, GPU에 model output인 logits 을 모두 저장하기 때문에 메모리 사용량이 너무 많은 문제 발생
    - 해결 방법
        - argmax method를 사용하여 logits에서 가장 의미 있는 큰 값만 뽑아내서 5만 개의 차원을 1 개로 축소.
        - logits[0].shape >>> (8, 55, 51865)
        - logits[0].argmax(dim=-1) >>> (8, 55, 1)

---

# 5. 결론

### 5.1. Related Research와 개선 정도 비교

- Training Time
    - 26시간 소요 with 8* NVIDIA A6000 GPU  - Related research
    - 2시간 소요 with 1* NVIDIA 4080 GPU  - Fine tuned whisper small model
- Using Dataset
    - 약 1000시간의 한국어 음성 데이터셋  - Related research
    - 약 33시간의 한국어 노인 음성 데이터셋  - Fine tuned whisper small model
- CER
    - 8.36%   - Related research’s fine-tuned whisper medium model
    - 11.39%   - Fine tuned whisper small model

### 5.2. Baseline model과 Performance 비교

- CER
    - 30.33% - Baseline whisper small model
    - 11.39%   - fine tuned whisper small model
- Example
    - 노인남여_노인대화09_F_1533066829_61_수도권_실내_10283.wav
        
        ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/6b3ed91b-8046-4da5-9275-98597568e3f1/77c3599c-847f-4543-80c5-8fe84732a08a/Untitled.png)
        
    - 노인남여_노인대화08_M_522912996_68_충청_실내_09064.wav
        
        ![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/6b3ed91b-8046-4da5-9275-98597568e3f1/46fa511e-0dd5-45a4-a445-7454f88e381a/Untitled.png)
        
- Conclusion
    - Baseline model과 비교하면 소리가 상대적으로 뭉개지는 노인 음성 데이터에 대하여
    한 글자 정도의 차이로 성능을 앞서는 것을 볼 수 있고, 실험이 성공하였음을 알 수 있다.
