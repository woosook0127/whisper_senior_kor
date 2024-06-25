'''Using very few samples due to preprocessing time(400H)
splited_senior_dataset
DatasetDict({
    train: Dataset({
        features: ['input_features', 'labels'],
        num_rows: 10715
    })
    test: Dataset({
        features: ['input_features', 'labels'],
        num_rows: 2679
    })
})

senior_dataset["validation"]
Dataset({
    features: ['input_features', 'labels'],
    num_rows: 9944
})
'''
# Set GPU before import torch.
# if not, "CUDA_VISIBLE_DEVICES" not working correctly
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
os.environ["CUDA_VISIBLE_DEVICES"]= "1"  # Set the GPU 1 to use
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

if (os.environ["CUDA_VISIBLE_DEVICES"]=="1"):
    import glob
    import datasets 
    from datasets import load_dataset
    from datasets import Features, Audio

    from transformers import WhisperTokenizer
    from transformers import WhisperFeatureExtractor
    from transformers import WhisperProcessor
    from transformers import WhisperForConditionalGeneration

    from dataclasses import dataclass
    from typing import Any, Dict, List, Union
    import evaluate

    from transformers import Seq2SeqTrainingArguments
    from transformers import Seq2SeqTrainer
    import torch
else:
    print(f"CUDA_VISIBLE_DEVICES is not 1")
    os._exit(1)

class WhisperTraining:
    def __init__(self):
        # set file path
        self.training_raw_path = "/data/freetalk_senior/1.Training/raw_data/1.AI챗봇/1.AI챗봇_2_자유대화(노인남여)_TRAINING/"
        self.training_labeled_path = "/data/freetalk_senior/1.Training/labeled_data/1.AI챗봇/1.AI챗봇_라벨링_자유대화(노인남여)_TRAINING/"
        self.training_dir = [
            "노인남여_노인대화12_M_1534748575_63_강원_실내",
            "노인남여_노인대화13_F_1527537495_69_강원_실내",
            "노인남여_노인대화12_M_1554478480_62_수도권_실내",
            "노인남여_노인대화13_F_1533490517_65_수도권_실내",
            "노인남여_노인대화13_M_1532565861_61_전라_실내",
            "노인남여_노인대화15_F_1535863537_62_전라_실내",
            "노인남여_노인대화13_F_1555169713_66_기타_실내",
            "노인남여_노인대화15_F_1540391606_71_기타_실내",
            "노인남여_노인대화14_M_1542738257_68_경상_실내",
            "노인남여_노인대화15_F_1534589663_68_경상_실내",
            "노인남여_노인대화14_M_1531684542_70_충청_실내",
            "노인남여_노인대화14_F_1534566769_66_충청_실내"
        ]
        #----------------------------------------------------------------------------------------------------------------
        self.validation_raw_path = "/data/freetalk_senior/2.Validation/raw_data/1.AI챗봇/1.AI챗봇_자유대화(노인남여)_VALIDATION"
        self.validation_labeled_path = "/data/freetalk_senior/2.Validation/labeled_data/1.AI챗봇/1.AI챗봇_라벨링_자유대화(노인남여)_VALIDATION"
        self.validation_dir = [
            "노인남여_노인대화07_M_1527639004_76_강원_실내",
            "노인남여_노인대화07_F_1526732595_65_강원_실내",
            "노인남여_노인대화07_M_1520916170_75_수도권_실내",
            "노인남여_노인대화07_F_1520511716_63_수도권_실내",
            "노인남여_노인대화12_M_1532565861_61_전라_실내",
            "노인남여_자유대화_F_1521502420_62_전라_실내",
            "노인남여_노인대화08_F_1524884567_60_기타_실내",
            "노인남여_노인대화07_M_1527671642_80_경상_실내",
            "노인남여_노인대화08_F_1529841959_62_경상_실내",
            "노인남여_노인대화08_M_1528952078_70_충청_실내",
            "노인남여_노인대화08_F_1527825984_71_충청_실내",
        ]
        print("SET: file path")

    def check_gpu(self):
        print(f"SYS: Using {os.environ.get('CUDA_VISIBLE_DEVICES')} devices")

    def load_data(self):
        self.training_raw_data = []
        training_raw_dir = [path + "/*.wav" for path in self.training_dir]
        for d in training_raw_dir:
            self.training_raw_data.append(sorted(glob.glob(os.path.join(self.training_raw_path, d), recursive=True)))

        self.training_labeled_data = []
        training_labeled_dir = [path + "/*.json" for path in self.training_dir]
        for d in training_labeled_dir:
            self.training_labeled_data.append(sorted(glob.glob(os.path.join(self.training_labeled_path, d), recursive=True)))
        #----------------------------------------------------------------------------------------------------------------
        self.validation_raw_data = []
        validation_raw_dir = [path + "/*.wav" for path in self.validation_dir]
        for d in validation_raw_dir:
            self.validation_raw_data.append(sorted(glob.glob(os.path.join(self.validation_raw_path, d), recursive=True)))

        self.validation_labeled_data = []
        validation_labeled_dir = [path + "/*.json" for path in self.validation_dir]
        for d in validation_labeled_dir:
            self.validation_labeled_data.append(sorted(glob.glob(os.path.join(self.validation_labeled_path, d), recursive=True)))
        #----------------------------------------------------------------------------------------------------------------
        print(f"LOAD: TRAINING:   raw:{sum(len(data) for data in self.training_raw_data)}, label:{sum(len(data) for data in self.training_labeled_data)}")
        print(f"LOAD: VALIDATION: raw:{sum(len(data) for data in self.validation_raw_data)}, label:{sum(len(data) for data in self.validation_labeled_data)}")

    def set_data(self):
        # setting DatasetDict for preprocessor input
        self.data_files = {
            "train_input": self.training_raw_data,    
            "train_label": self.training_labeled_data,
            "validation_input": self.validation_raw_data,    
            "validation_label": self.validation_labeled_data,
        }

        # load_dataset 이 자동으로 만드는 label column 때문에 에러 발생하여 features명시
        self.features = Features({
            'audio': Audio()
        })

        # Set Training data
        train_input = []
        for d in self.data_files['train_input']:
            train_input.append(load_dataset("audiofolder", 
                                data_files=d, 
                                features=self.features)
                            )
        train_input = [x['train'] for x in train_input]
        train_input = datasets.DatasetDict({
            "train": datasets.concatenate_datasets(train_input, axis=0)
        })
        #----------------------------------------------------------------------------------------------------------------
        train_label = []
        for d in self.data_files['train_label']:
            train_label.append(load_dataset("json", data_files=d)
                            )
        train_label = [x['train'] for x in train_label]
        train_label = datasets.DatasetDict({
            "train": datasets.concatenate_datasets(train_label, axis=0)
        })
        #----------------------------------------------------------------------------------------------------------------
        # Set Validataion data
        validation_input = []
        for d in self.data_files['validation_input']:
            validation_input.append(load_dataset("audiofolder", 
                                data_files=d, 
                                features=self.features)
                            )
        validation_input = [x['train'] for x in validation_input]
        validation_input = datasets.DatasetDict({
            "validation": datasets.concatenate_datasets(validation_input, axis=0)
        })
        #----------------------------------------------------------------------------------------------------------------
        validation_label = []
        for d in self.data_files['validation_label']:
            validation_label.append(load_dataset("json", data_files=d))
            
        validation_label = [x['train'] for x in validation_label]
        validation_label = datasets.DatasetDict({
            "validation": datasets.concatenate_datasets(validation_label, axis=0)
        })
        #----------------------------------------------------------------------------------------------------------------
        
        # merged dataset
        self.senior_dataset = datasets.DatasetDict()
        self.senior_dataset["train"] =datasets.concatenate_datasets([train_input['train'], 
                                                                     train_label['train']], axis=1)
        self.senior_dataset["validation"] = datasets.concatenate_datasets([validation_input['validation'], 
                                                                           validation_label['validation']], axis=1)
        # final dataset form
        self.senior_dataset = self.senior_dataset.shuffle(seed=44)
        print("SET: senior dataset")

    def set_preprocessor(self):
        self.checkpoint = "jiwon65/whisper-small_korean-zeroth"

        # Whisper model output: vocabulary 단어의 index
        # 이를 text format으로 postprocessing 하기 위해 Tokenizer 사용
        self.tokenizer = WhisperTokenizer.from_pretrained(self.checkpoint, language="Korean", task="transcribe")

        # audio sample을 log-mel spectogram으로 변환
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(self.checkpoint)
        
        # 위 module을 하나로 묶는다
        self.processor = WhisperProcessor.from_pretrained(self.checkpoint, language="Korean", task="transcribe")
        print("SET: preprocessors")

    def run_preprocessing(self):
        def preprocess_dataset(batch):
            audio=batch["audio"]
            label=batch["발화정보"]
            
            batch["input_features"] = self.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

            batch["labels"] = self.tokenizer(label['stt']).input_ids
            return batch

        self.senior_dataset = self.senior_dataset.map(preprocess_dataset, 
                                                      remove_columns=self.senior_dataset.column_names["train"], 
                                        )
        self.splited_senior_dataset = self.senior_dataset["train"].train_test_split(test_size=0.2)
        print("DONE: preprocessing senior dataset")
        
    def set_datacollator(self):
        # DatasetDict -> torch.tensor   
        @dataclass
        class DataCollatorSpeechSeq2SeqWithPadding:
            processor: Any
            
            def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
                # 인풋 데이터와 라벨 데이터의 길이가 다르며, 따라서 서로 다른 패딩 방법이 적용되어야 한다. 그러므로 두 데이터를 분리해야 한다.
                # 먼저 오디오 인풋 데이터를 간단히 토치 텐서로 반환하는 작업을 수행한다.
                input_features = [{"input_features": feature["input_features"]} for feature in features]
                batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
                
                # Tokenize된 레이블 시퀀스를 가져온다.
                label_features = [{"input_ids": feature["labels"]} for feature in features]
                # 레이블 시퀀스에 대해 최대 길이만큼 패딩 작업을 실시한다.
                labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
                
                # 패딩 토큰을 -100으로 치환하여 loss 계산 과정에서 무시되도록 한다.
                labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
                
                # 이전 토크나이즈 과정에서 bos 토큰이 추가되었다면 bos 토큰을 잘라낸다.
                # 해당 토큰은 이후 언제든 추가할 수 있다.
                if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
                    labels = labels[:, 1:]
                
                batch["labels"] = labels
                
                return batch

        self.data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=self.processor)
        print("SET: datacollator")
        
    def set_metric(self):
        self.metric = evaluate.load('cer')
        print("SET: metric = cer")

    def compute_metrics(self, pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        label_ids[label_ids == -100] = self.tokenizer.pad_token_id

        pred_str = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = self.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        cer = 100 * self.metric.compute(predictions=pred_str, references=label_str)

        return {"cer": cer}

    def preprocess_logits_for_metrics(self, logits, labels):
        pred_ids = logits[0].argmax(dim=-1)
        return pred_ids

    def set_model(self):
        self.model = WhisperForConditionalGeneration.from_pretrained(self.checkpoint)
        self.model.config.forced_decoder_ids = None
        self.model.config.suppress_tokens = []
        self.model.config.use_cache = False # mem usage 줄이기 위함
        print("SET: model")

    def set_gpu(self):
        print("Number of GPUs available:", torch.cuda.device_count())
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.model = self.model.to(device)

    def set_trainer(self):
        steps = 400
        self.training_args = Seq2SeqTrainingArguments(
           output_dir="./train_result_last",       # output dir
            # logging_dir="./logging_dir",       # storing to log_dir
            max_steps=steps,                   # epoch 대신 설정
            # num_train_epochs = 5,              # total num of training epochs
            evaluation_strategy="steps",       # evaluate after each "steps"

            warmup_steps=int(steps/20),             # num of warmup steps for learning rate scheduler
            save_steps=int(steps/8),
            eval_steps=int(steps/8),
            logging_steps=int(steps/20),            # how often to print log
            
            per_device_train_batch_size=64,    # batch size of per_device during 
            per_device_eval_batch_size=32,      # batch size per GPU for evaluation
            # gradient_accumulation_steps=2,    # tot num of steps before back propagation. train time increase.
            learning_rate=1e-5,                # initial lr for AdamW optimizer
            weight_decay=0.01,                 # strength of weight decay
            gradient_checkpointing=True,       # saving mem, but make backward pass slower
            fp16=True,                         # use mixed precision
            # fp16_opt_level="02",               # mixed precision mode
            
            # predict_with_generate=True,        # using generate to calculate 'generative metrics'(ROUGE, BLEU)
            # generation_max_length=225,         # generative metrics's max len

            report_to=["tensorboard"],
            load_best_model_at_end=True,       # best checkpoint will always be saved
            
            metric_for_best_model="cer",       # 한국어의 경우 띄어쓰기가 애매한 경우가 많아서 'wer'보다는 'cer'이 더 적합할 것
            greater_is_better=False,           # is better metric result is big?
            push_to_hub=False,
            # do_train=True,                   # Perform training
            # do_eval=True,                    # Perform evaluation
            run_name="TrainingFinal",          # experiment name
            # seed=3,                            # seed for experiment reproductbility 3x3
        )
        print("SET: training args")

        self.trainer = Seq2SeqTrainer(
            args=self.training_args,
            model=self.model,
            train_dataset=self.splited_senior_dataset["train"],
            eval_dataset=self.splited_senior_dataset["test"],
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics,
            tokenizer=self.processor.feature_extractor,
            preprocess_logits_for_metrics=self.preprocess_logits_for_metrics,
        )
        print("SET: trainer")

    def run_training(self):
        torch.cuda.empty_cache()
        self.trainer.train()

    def run(self):
        self.check_gpu()
        self.load_data()
        self.set_data()
        self.set_preprocessor()
        self.run_preprocessing()
        self.set_datacollator()
        self.set_metric()
        self.set_model()
        self.set_gpu()
        self.set_trainer()
        
        self.run_training()

if __name__ == "__main__":
    exec = WhisperTraining()

    exec.run()