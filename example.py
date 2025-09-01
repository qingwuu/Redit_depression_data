#适合全量数据、低内存
#不一次性读入全文件，逐块统计，轻量可视化

from collections import defaultdict
from pathlib import Path
from pydoc import resolve
import re
from tkinter import OUTSIDE
import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
from io import StringIO


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
total_words=0
#set:自动去重
#使用：lowercase，【unknown】--->len（author）
authors=set()
sum_words=0
sum_chars=0
date_min=None
date_max=None
#Counter:Counter是计数器字典
month_counts=Counter()
word_counts=Counter()

#TOKEN_RE.findall(text)#TOKEN_RE.findall(text)
HIST_BIN=50
hist_bins=defaultdict(int)

#MAIN
for chunk in pd.read_csv(CSV_PATH, chunksize=CHUNK_SIZE,dtype=str):
    for c in ["author","created_utc",'selftext']:
        if c not in chunk.columns:
            chunk[c]=""

    st=chunk["selftext"].fillna("")
    for txt in st:
        for t in tokenize(txt):
            word_counts[t]+=1
    authors.update(chunk["author"].fillna("[unknown]").str.lower().tolist())
    
    words_per_row=st.str.split().map(len).astype(int)
    chars_per_row=st.str.len().astype(int)
    for wc in words_per_row:
        bin_id=(wc//HIST_BIN)*HIST_BIN
        hist_bins[bin_id]+=1


    total_rows+=len(chunk)
    
    sum_words+=int(words_per_row.sum())
    sum_chars+=int(chars_per_row.sum())


    cu=chunk["created_utc"].fillna("")
    dt_epoch=pd.to_numeric(cu,errors="coerce")
    dt=pd.to_datetime(dt_epoch,unit="s",utc=True,errors="coerce")
    need_fallback=dt.isna()
    if need_fallback.any():
        dt.loc[need_fallback]=pd.to_datetime(cu[need_fallback],utc=True,errors="coerce")

    if dt.notna().any():
        dmin=dt.min()
        dmax=dt.max()
        date_min=dmin if date_min is None or dmin<date_min else date_min
        date_max=dmax if date_max is None or dmax>date_max else date_max
        month_counts.update(dt.dt.to_period("M").astype(str).value_counts().to_dict())

    avg_words=(sum_words/total_rows) if total_rows else 0.0
    avg_chars=(sum_chars/total_rows) if total_rows else 0.0

    metrics=pd.DataFrame([{
        "total_posts":total_rows,
        "unique_authors":len(authors),
        "avg_words_per_post":round(avg_words,2),
        "avg_chars_per_post":round(avg_chars,2),
        "date_min_utc":("" if date_min is None else str(date_min)),
        "date_max_utc":("" if date_max is None else str(date_max))
    }])
    metrics.to_csv(OUTDIR/"metrics_summary.csv",index=False)

    top20=pd.DataFrame(word_counts.most_common(20),columns=["word","frequency"])
    top20.to_csv(OUTDIR/"top_20_words.csv",index=False)



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

    print("done, dir:",OUTDIR.resolve())
