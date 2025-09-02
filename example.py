#适合全量数据、低内存
#不一次性读入全文件，逐块统计，轻量可视化

from collections import defaultdict
from pathlib import Path
import random
import re
import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
from io import StringIO
from sklearn.feature_extraction.text import TfidfVectorizer

CSV_PATH="depression-sampled.csv"
OUTDIR=Path("chang_shen_result")
#mkdie(exit_ok=True):如果文件夹已经存在就别报错，直接继续（很常用的防御性写法）
OUTDIR.mkdir(exist_ok=True)


CHUNK_SIZE=100000
TOKEN_RE=re.compile(r"\b[a-zA-Z]{3,}\b")


# 为什么用set：
# 集合自动去重
# 用于成员测试快
STOP=set(pd.Series(pd.read_csv(
    StringIO("word\n"+"\n".join([
        #credit:https://gist.github.com/sebleier/554280
        "i","me","my","myself","we","our","ours","ourselves","you","your","yours","yourself","yourselves","he","him","his","himself","she","her","hers","herself","it","its","itself","they","them","their","theirs","themselves","what","which","who","whom","this","that","these","those","am","is","are","was","were","be","been","being","have","has","had","having","do","does","did","doing","a","an","the","and","but","if","or","because","as","until","while","of","at","by","for","with","about","against","between","into","through","during","before","after","above","below","to","from","up","down","in","out","on","off","over","under","again","further","then","once","here","there","when","where","why","how","all","any","both","each","few","more","most","other","some","such","no","nor","not","only","own","same","so","than","too","very","s","t","can","will","just","don","should","now"
        #some words that are in top20 but i think those were not helpful to research purpose so add them into stop words
        ,"like","want","get","things","even","really","one","would","much","going","things"
        ,"feel","know","never","day","years","anything","always","back","make","still","everything","something"
        ,"could","way","every","got","anyone"
    ]#python list,contains frequently stop words
    )#把这个list拼成一个CSV格式的字符串。第一行是title ”word",后面每一行一个单词
    )#把string包装成object文件对象，让pd.read_csv(...)直接从内存中的字符串读取
)['word']).tolist()) #从DataFrame里取出word这一列，得到一个series，把series变成一个普通的python列表

# STOP={"i","me","my","myself","we","our","ours","ourselves","you","your","yours","yourself","yourselves","he","him","his","himself","she","her","hers","herself","it","its","itself","they","them","their","theirs","themselves","what","which","who","whom","this","that","these","those","am","is","are","was","were","be","been","being","have","has","had","having","do","does","did","doing","a","an","the","and","but","if","or","because","as","until","while","of","at","by","for","with","about","against","between","into","through","during","before","after","above","below","to","from","up","down","in","out","on","off","over","under","again","further","then","once","here","there","when","where","why","how","all","any","both","each","few","more","most","other","some","such","no","nor","not","only","own","same","so","than","too","very","s","t","can","will","just","don","should","now"}


def tokenize(text):
    #如果text不是string/是空string，直接返回空列表
    if not isinstance(text,str) or not text:
        return []
    return [t.lower() for t in TOKEN_RE.findall(text) if t.lower() not in STOP]
#TOKEN_RE.findall(text)会返回一个列表
#列表所有的词都转为小写，不在stop words里的保留

total_rows=0
#set:自动去重
#使用：lowercase，【unknown】--->len（author）
authors=set()
sum_words=0
sum_chars=0
date_min=None
date_max=None
#Counter:Counter是计数器字典
month_counts=Counter()

#EXTRA:MN local time
month_counts_mn=Counter()
word_counts=Counter()

#TOKEN_RE.findall(text)#TOKEN_RE.findall(text)
HIST_BIN=50
hist_bins=defaultdict(int)

#EXTRA:TF-IDF,突出相对独特的词
_rng=random.Random(42)
sample_texts=[]
seen=0

#MAIN
for chunk in pd.read_csv(CSV_PATH, chunksize=CHUNK_SIZE,dtype=str):
    for c in ["author","created_utc","selftext","title"]:
        if c not in chunk.columns:
            chunk[c]=""

    #EXTRA:有可能有些有用信息在title里面，把title也添加到分析文本中
    st=(chunk["title"].fillna("")+" "+chunk["selftext"].fillna("")).astype(str)

    for txt in st:
        for t in tokenize(txt):
            word_counts[t]+=1

    #EXTRA:过滤无效作者
    ignore_authors={"automoderator","[deleted]","[unknown]"}
    authors.update(chunk["author"].fillna("[unknown]").str.lower().pipe(lambda s:s[~s.isin(ignore_authors)]).tolist())
    
    words_per_row=st.str.split().map(len).fillna(0).astype(int)
    chars_per_row=st.str.len().fillna(0).astype(int)
    for wc in words_per_row:
        bin_id=(wc//HIST_BIN)*HIST_BIN
        hist_bins[bin_id]+=1


    total_rows+=len(st)
    
    sum_words+=int(words_per_row.sum())
    sum_chars+=int(chars_per_row.sum())


    cu=chunk["created_utc"].fillna("")
    dt_utc=pd.to_numeric(cu,errors="coerce")
    dt=pd.to_datetime(dt_utc,unit="s",utc=True,errors="coerce")
    need_fallback=dt.isna()
    if need_fallback.any():
        dt.loc[need_fallback]=pd.to_datetime(cu[need_fallback],utc=True,errors="coerce")

    
    if dt.notna().any():
        dmin=dt.min()
        dmax=dt.max()
        date_min=dmin if date_min is None or dmin<date_min else date_min
        date_max=dmax if date_max is None or dmax>date_max else date_max
        month_counts.update(dt.dt.to_period("M").astype(str).value_counts().to_dict())
        #EXTRA:date in minneapolis time!
        dt_mn=dt.dt.tz_convert("America/Chicago")
        month_mn=dt_mn.dt.to_period("M").astype(str).value_counts().to_dict()
        month_counts_mn.update(month_mn)


    #EXTRA:TF-IDF,突出相对独特的词
    for txt in st:
        if not txt:
            continue
        seen+=1
        if len(sample_texts)<10000:
            sample_texts.append(txt)
        else:
            j=_rng.randint(1,seen)
            if j<=10000:
                sample_texts[j-1]=txt
if sample_texts:
    vec=TfidfVectorizer(stop_words="english",token_pattern=r"(?u)\b[a-zA-Z]{3,}\b",max_features=20000)
    X=vec.fit_transform(sample_texts)
    scores=X.sum(axis=0).A1
    terms=vec.get_feature_names_out()
    idx=scores.argsort()[::-1][:20]
    tfidf_top20=pd.DataFrame({"word":terms[idx],"tfidf_score":scores[idx]})
    tfidf_top20.to_csv(OUTDIR/"tfidf_top20.csv",index=False)


avg_words=(sum_words/total_rows) if total_rows else 0.0
avg_chars=(sum_chars/total_rows) if total_rows else 0.0



top20=pd.DataFrame(word_counts.most_common(20),columns=["word","frequency"])
top20.to_csv(OUTDIR/"top_20_words.csv",index=False)

#EXTRA：指标更稳健：加入中位数、分位数、极端值裁剪
def approx_quantile_from_hist(hist:dict,binw:int,q:float)->float:
    if not hist:
        return 0.0
    total=sum(hist.values())
    target=total*q
    acc=0
    for start in sorted(hist.keys()):
        acc+=hist[start]
        if acc>=target:
            return start+binw/2.0
    last=max(hist.keys())
    return last+binw/2.0
median_words=approx_quantile_from_hist(hist_bins,HIST_BIN,0.5)
p90_words=approx_quantile_from_hist(hist_bins,HIST_BIN,0.9)

metrics=pd.DataFrame([{
    "total_posts":total_rows,
    "unique_authors":len(authors),
    "avg_words_per_post":round(avg_words,2),
    "avg_chars_per_post":round(avg_chars,2),
    "date_min_utc":("" if date_min is None else str(date_min)),
    "date_max_utc":("" if date_max is None else str(date_max)),
    "median_words_per_post":round(median_words,2),
    "p90_words_per_post":round(p90_words,2),
}])
metrics.to_csv(OUTDIR/"metrics_summary.csv",index=False)

if month_counts:
    mc=pd.Series(month_counts).sort_index()
    plt.figure()
    mc.plot(kind="line",marker="o")
    plt.title("post per month (UTC)")
    plt.xlabel("month")
    plt.ylabel("posts")
    plt.tight_layout()
    plt.savefig(OUTDIR/"posts_per_month.png")
    plt.close()

if month_counts_mn:
    mc_mn = pd.Series(month_counts_mn).sort_index()
    plt.figure()
    mc_mn.plot(kind="line", marker="o")
    plt.title("post per month (America/Chicago)")
    plt.xlabel("month")
    plt.ylabel("posts")
    plt.tight_layout()
    plt.savefig(OUTDIR/"posts_per_month_mn.png")
    plt.close()

if hist_bins:
    hb=pd.Series(hist_bins).sort_index()
    plt.figure()
    plt.bar(hb.index.astype(int),hb.values.astype(int),width=HIST_BIN*0.9)
    plt.title("word count distribution")
    plt.xlabel(f"word count (bin={HIST_BIN})")
    plt.ylabel("posts")
    plt.tight_layout()
    plt.savefig(OUTDIR/"word_count_hist.png")
    plt.close()

if not top20.empty:
    plt.figure()
    idx=np.arange(len(top20))
    plt.bar(idx,top20["frequency"].values.astype(int))
    plt.xticks(idx,top20["word"].tolist(),rotation=75)
    plt.title("top 20 words")
    plt.xlabel("word")
    plt.ylabel("frequency")
    plt.tight_layout()
    plt.savefig(OUTDIR/"top20_words.png")
    plt.close()

    
# #EXTRA:用pandas.cut生成箱
# bins=list(range(0,int(words_per_row.max())+51,50))
# cats=pd.cut(words_per_row,bins=bins,right=False)
# hist_counts=cats.value_counts().sort_index()



print("DONE!!!, dir:",OUTDIR.resolve())
