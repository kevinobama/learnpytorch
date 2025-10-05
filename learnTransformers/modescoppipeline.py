from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

nlp_pipeline = pipeline(task=Tasks.text_generation, model='damo/gpt2-text-generation')
result = nlp_pipeline(input="Once upon a time, in a land far away,")
print(result)