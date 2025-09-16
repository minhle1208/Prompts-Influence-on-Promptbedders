В работе изучается влияние формулировки и языка промптов на качество плотностного поиска (dense retrieval) для русскоязычных запросов

Разработан и воспроизведён pipeline “перевод → генерация → автофильтрация → обучение" 

1. В файле [Translate msmarco.py](https://github.com/minhle1208/Prompts-Influence-on-Promptbedders-/blob/main/Translate%20msmarco.py) переведен набор данных MS MARCO(tevatron-msmarco-aug) на русский язык с помощью Yandex translate API
2. [kursovaya-dataset-w-instructions-855-next-500(final).ipynb](https://github.com/minhle1208/Prompts-Influence-on-Promptbedders-/blob/main/kursovaya-dataset-w-instructions-855-next-500(final).ipynb): Генерированы инструкции с помощью
RefalMachine/RuadaptQwen2.5-7B-Lite-Beta(В этой версией есть код не для всех выборок, так так вынуждено срезать датасет на несколько штук чтобы быстрее генерировать, но можно служить как пример для повторения)
3. [dataset_w_inst_ru_filtered.ipynb](https://github.com/minhle1208/Prompts-Influence-on-Promptbedders-/blob/main/dataset_w_inst_ru_filtered.ipynb): Оценена актуальность инструкций с помощью ai-forever/sbert_large_nlu_ru+косинусная близость и отфильтровал
4. [instructions_one_best_per_id.py](https://github.com/minhle1208/Prompts-Influence-on-Promptbedders-/blob/main/instructions_one_best_per_id.py): Выбрана лучшая инструкция для каждого row_id, которая имеет наилучший показатель дельта по сравнению с другими инструкциями в row_id(датасет можно смотреть в файле [instructions_one_best_per_id.jsonl](https://github.com/minhle1208/Prompts-Influence-on-Promptbedders/blob/main/instructions_one_best_per_id.jsonl)
5. [model_train and tests.ipynb](https://github.com/minhle1208/Prompts-Influence-on-Promptbedders-/blob/main/model_train%20and%20tests.ipynb): Обучена модель на базе intfloat/multilingual-e5-base(LoRA + InfoNCE) и протестировал на робастность
