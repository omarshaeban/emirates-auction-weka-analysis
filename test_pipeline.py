import pandas as pd
from scraper import parse_html,save_csv
from prepare_data import prepare,duration_minutes
from export_arff import to_arff
from compare_models import compare


def test_original_structure_and_arff(tmp_path):
    html='''<div class="list-card-container"><h2 id="title_123">تويوتا كامري 2002</h2>
    <div>التسلسل 123</div><div>الوقت المتبقي: 2 ساعة 15 دقيقة</div>
    <div>مزايدات: 8</div><div id="CARD_PRICE_123"><span class="text-decoration-line-through">5000</span><span>6,100 ر.س</span></div></div>'''
    r=parse_html(html)
    assert len(r)==1 and r[0]['price']=='6100' and r[0]['bids']=='8'
    assert duration_minutes(r[0]['time_left'])==135
    csv=tmp_path/'test.csv';save_csv(r,csv)
    assert 'تويوتا' in csv.read_text(encoding='utf-8-sig')


def test_weka_export_and_models():
    raw=pd.DataFrame([{'title':('تويوتا كامري' if i%3 else 'نيسان صني'), 'sequence':str(i),
                       'time_left':f'{(i%4)+1} ساعة', 'bids':str(i%8+1), 'price':str(10000+i*100)} for i in range(60)])
    frame=prepare(raw)
    arff=to_arff(frame)
    assert '@relation auction_high_price' in arff and '@attribute high_price {low,high}' in arff
    out=compare(frame,folds=3)
    assert out.shape[0]==9 and out.accuracy.between(0,1).all()
