import glob
import json



from preprocessing import PreProcessor
from textrank import TextRank

document_preprocessor = PreProcessor()
json_articles_path = 'data-source'
summaries_path = 'output/textrank'
articles = glob.glob(json_articles_path + '/**/*.json')
model = TextRank()
print(f'Found {len(articles)} articles')
for art in articles:
    print('Processing {}...'.format(art), end='\r')
    out_path = summaries_path + art.replace(json_articles_path,'').replace('.json','-summary.txt')
    with open(art, 'r', encoding='utf-8') as inf, open(out_path, 'w', encoding='utf-8') as outf:
        document_record = json.load(inf)
        segments = document_preprocessor.extract_segments(document_record)
        document_text = ". ".join(segments)
        summary =model.summarize(text=document_text,  ratio=0.18)
        if len(summary):
            outf.write(summary)
        else:
            print("Skipping empty summary")