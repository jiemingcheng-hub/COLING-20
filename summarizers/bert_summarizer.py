import glob
import json

from preprocessing import PreProcessor
from summarizers.bertsum import BertSummarizer

document_preprocessor = PreProcessor()
model = BertSummarizer(model='bert-base-multilingual-cased', reduce_option='max')
json_articles_path = 'data-source'
summaries_path = 'output/bert'
articles = glob.glob(json_articles_path + '/**/*.json')
print(f'Found {len(articles)} articles')
for art in articles:
    print('Processing {}...'.format(art), end='\r')
    out_path = summaries_path + art.replace(json_articles_path,'').replace('.json','-summary.txt')
    with open(art, 'r', encoding='utf-8') as inf, open(out_path, 'w', encoding='utf-8') as outf:
        document_record = json.load(inf)
        segments = document_preprocessor.extract_segments(document_record)
        document_text = ". ".join(segments)
        summary = model(document_text, min_length=0, max_length=500, ratio=0.18)
        if len(summary):
            outf.write(summary)
        else:
            print("Skipping empty summary")

