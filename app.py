import io, zipfile
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="RetailIQ | Online Retail Analytics", page_icon="📊", layout="wide")
st.markdown('''<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif}.stApp{background:#f5f7fb}.block-container{max-width:1450px;padding-top:1.5rem}
[data-testid="stSidebar"]{background:#101b35}[data-testid="stSidebar"] *{color:#edf3ff!important}
.hero{background:linear-gradient(115deg,#101b35,#244776 65%,#287c86);padding:28px 32px;border-radius:18px;color:white;margin-bottom:22px}.hero h1{font:800 2rem Manrope;margin:0;color:white}.hero p{color:#d8e5f8;margin:6px 0 0}
.metric{background:white;border:1px solid #e8edf5;border-radius:14px;padding:17px 19px;box-shadow:0 3px 12px #1a2b4d09}.metric-label{color:#71809a;font-size:.75rem;font-weight:700;text-transform:uppercase;letter-spacing:.07em}.metric-value{color:#142443;font:800 1.55rem Manrope;margin-top:5px}.section{font:800 1.2rem Manrope;color:#162746;margin:16px 0 10px}.note{background:#eaf2ff;border-left:4px solid #3977c8;border-radius:8px;padding:12px 15px;color:#29466f}
</style>''',unsafe_allow_html=True)

REQUIRED=['InvoiceNo','StockCode','Description','Quantity','InvoiceDate','UnitPrice','CustomerID','Country']
FEATURES=['TotalOrders','TotalQuantity','TotalRevenue','AverageOrderValue','AverageUnitPrice','Recency']

def load_file(data,name):
    if name.lower().endswith('.zip'):
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            files=[x for x in z.namelist() if x.lower().endswith('.csv') and not x.startswith('__MACOSX/')]
            if not files: raise ValueError('No CSV file found inside ZIP.')
            return pd.read_csv(z.open(files[0]),encoding='latin1')
    return pd.read_csv(io.BytesIO(data),encoding='latin1')

@st.cache_data(show_spinner='Preparing data warehouse and churn model…')
def analyse(raw):
    missing=[c for c in REQUIRED if c not in raw.columns]
    if missing: raise ValueError('Missing required columns: '+', '.join(missing))
    df=raw.copy(); df.columns=df.columns.str.strip(); original=len(df); duplicates=int(df.duplicated().sum())
    df=df.drop_duplicates().dropna(subset=['StockCode'])
    df['Description']=df['Description'].fillna('Unknown Product'); df['Country']=df['Country'].fillna('Unknown Country')
    df['CustomerID']=pd.to_numeric(df['CustomerID'],errors='coerce').astype('Int64')
    df['Quantity']=pd.to_numeric(df['Quantity'],errors='coerce'); df['UnitPrice']=pd.to_numeric(df['UnitPrice'],errors='coerce')
    df['InvoiceDate']=pd.to_datetime(df['InvoiceDate'],errors='coerce'); df=df.dropna(subset=['InvoiceDate'])
    df['InvoiceNo']=df['InvoiceNo'].astype(str); df['IsCancelled']=df['InvoiceNo'].str.startswith('C')
    sales=df[(~df.IsCancelled)&(df.Quantity>0)&(df.UnitPrice>0)].copy()
    if sales.empty: raise ValueError('No valid sales after cleaning.')
    sales['TotalAmount']=sales.Quantity* sales.UnitPrice
    for col,func in [('Year',lambda x:x.dt.year),('Month',lambda x:x.dt.month),('MonthName',lambda x:x.dt.month_name()),('Day',lambda x:x.dt.day),('DayName',lambda x:x.dt.day_name()),('Hour',lambda x:x.dt.hour)]: sales[col]=func(sales.InvoiceDate)
    cust=sales[['CustomerID','Country']].drop_duplicates().dropna(subset=['CustomerID']).drop_duplicates('CustomerID')
    cm=sales.groupby('CustomerID').agg(TotalOrders=('InvoiceNo','nunique'),TotalQuantity=('Quantity','sum'),TotalRevenue=('TotalAmount','sum'),AverageOrderValue=('TotalAmount','mean'),LastPurchaseDate=('InvoiceDate','max')).reset_index()
    cm['PurchaseFrequency']=cm.TotalOrders; cm['AverageQuantityPerOrder']=cm.TotalQuantity/cm.TotalOrders.replace(0,np.nan)
    ref=sales.InvoiceDate.max(); cm['Recency']=(ref-cm.LastPurchaseDate).dt.days; cust=cust.merge(cm,on='CustomerID',how='left').reset_index(drop=True); cust['CustomerKey']=range(1,len(cust)+1)
    prod=sales[['StockCode','Description','UnitPrice']].drop_duplicates('StockCode')
    pm=sales.groupby('StockCode').agg(TotalUnitsSold=('Quantity','sum'),TotalRevenue=('TotalAmount','sum'),TotalTransactions=('InvoiceNo','nunique'),AverageSellingPrice=('UnitPrice','mean')).reset_index()
    prod=prod.merge(pm,on='StockCode',how='left').reset_index(drop=True); prod['ProductKey']=range(1,len(prod)+1)
    date=sales[['InvoiceDate','Year','Month','MonthName','Day','DayName','Hour']].copy(); date['DateKey']=date.InvoiceDate.dt.strftime('%Y%m%d').astype(int); date=date.drop_duplicates('DateKey').reset_index(drop=True)
    fact=sales[['InvoiceNo','CustomerID','StockCode','InvoiceDate','Year','Month','MonthName','Day','DayName','Hour','Quantity','UnitPrice','TotalAmount']].copy()
    fact['CustomerKey']=fact.CustomerID.map(cust.set_index('CustomerID').CustomerKey); fact['ProductKey']=fact.StockCode.map(prod.set_index('StockCode').ProductKey); fact['DateKey']=fact.InvoiceDate.dt.strftime('%Y%m%d').astype(int)
    fact=fact.dropna(subset=['CustomerKey','ProductKey','DateKey'])
    for c in ['CustomerKey','ProductKey','DateKey']: fact[c]=fact[c].astype(int)
    # The notebook's 90-day historical window and future churn label.
    cutoff=ref-pd.Timedelta(days=90); hist=sales[(sales.InvoiceDate<cutoff)&sales.CustomerID.notna()]; future=sales[(sales.InvoiceDate>=cutoff)&(sales.InvoiceDate<=ref)&sales.CustomerID.notna()]
    churn=hist.groupby('CustomerID').agg(TotalOrders=('InvoiceNo','nunique'),TotalQuantity=('Quantity','sum'),TotalRevenue=('TotalAmount','sum'),AverageOrderValue=('TotalAmount','mean'),AverageUnitPrice=('UnitPrice','mean'),LastPurchaseDate=('InvoiceDate','max')).reset_index()
    churn['Recency']=(cutoff-churn.LastPurchaseDate).dt.days
    f=future.groupby('CustomerID').agg(FutureOrders=('InvoiceNo','nunique')).reset_index(); churn=churn.merge(f,on='CustomerID',how='left'); churn['FutureOrders']=churn.FutureOrders.fillna(0); churn['Churn']=(churn.FutureOrders==0).astype(int)
    metrics=None; model=None
    if len(churn)>=20 and churn.Churn.nunique()==2:
        Xtr,Xte,ytr,yte=train_test_split(churn[FEATURES],churn.Churn,test_size=.2,random_state=42,stratify=churn.Churn)
        model=RandomForestClassifier(n_estimators=200,max_depth=10,min_samples_split=5,min_samples_leaf=2,random_state=42,class_weight='balanced'); model.fit(Xtr,ytr)
        pred=model.predict(Xte); prob=model.predict_proba(Xte)[:,1]
        metrics={'Accuracy':accuracy_score(yte,pred),'Precision':precision_score(yte,pred,zero_division=0),'Recall':recall_score(yte,pred,zero_division=0),'F1':f1_score(yte,pred,zero_division=0),'ROC-AUC':roc_auc_score(yte,prob)}
    return dict(raw=raw,sales=sales,fact=fact,customer=cust,products=prod,date=date,churn=churn,metrics=metrics,model=model,original=original,duplicates=duplicates,cancelled=int(df.IsCancelled.sum()),valid=len(sales),reference=ref,cutoff=cutoff)

def metric(label,value):
    st.markdown(f'<div class="metric"><div class="metric-label">{label}</div><div class="metric-value">{value}</div></div>',unsafe_allow_html=True)
def chart(fig):
    fig.update_layout(paper_bgcolor='white',plot_bgcolor='white',margin=dict(l=10,r=10,t=48,b=10))
    st.plotly_chart(fig,use_container_width=True)

st.sidebar.markdown('## 📊 RetailIQ'); st.sidebar.caption('Data Mining & Data Warehouse Analytics')
up=st.sidebar.file_uploader('Upload Online Retail dataset',type=['csv','zip'],help='Use data.csv or data.csv.zip from your Google Drive folder.')
st.markdown('<div class="hero"><h1>Online Retail Intelligence</h1><p>Sales analytics · OLAP warehouse · Customer churn prediction</p></div>',unsafe_allow_html=True)
if up is None:
    st.info('Upload the CSV or ZIP file in the left sidebar to load the interactive dashboard.')
    st.markdown('### Required dataset columns'); st.write(', '.join(REQUIRED))
    st.markdown('<div class="note"><b>Dataset location in your notebook:</b> Google Drive → <code>data mining and warehouse/data.csv.zip</code>. The notebook does not contain the CSV itself, so download that file and upload it here.</div>',unsafe_allow_html=True)
    st.stop()
try: d=analyse(load_file(up.getvalue(),up.name))
except Exception as e: st.error(f'Unable to process dataset: {e}'); st.stop()
sales,fact,cust,prod,churn=d['sales'],d['fact'],d['customer'],d['products'],d['churn']
st.sidebar.markdown('### Filters'); years=st.sidebar.multiselect('Year',sorted(sales.Year.unique()),default=sorted(sales.Year.unique())); countries=st.sidebar.multiselect('Country',sorted(sales.Country.dropna().unique()))
view=sales[sales.Year.isin(years)]; view=view[view.Country.isin(countries)] if countries else view
if view.empty: st.warning('No data for these filters.'); st.stop()
cols=st.columns(5)
for col,(label,val) in zip(cols,[('Net revenue',f'£{view.TotalAmount.sum():,.2f}'),('Transactions',f'{view.InvoiceNo.nunique():,}'),('Units sold',f'{view.Quantity.sum():,.0f}'),('Customers',f'{view.CustomerID.nunique():,}'),('Products',f'{view.StockCode.nunique():,}')]):
    with col: metric(label,val)

t1,t2,t3,t4,t5,t6=st.tabs(['Overview','Products & Countries','Customers','Warehouse & OLAP','Churn Model','Data Quality'])
with t1:
    st.markdown('<div class="section">Sales performance</div>',unsafe_allow_html=True)
    monthly=view.groupby(pd.Grouper(key='InvoiceDate',freq='MS')).agg(Revenue=('TotalAmount','sum')).reset_index(); a,b=st.columns([1.6,1])
    with a: chart(px.area(monthly,x='InvoiceDate',y='Revenue',markers=True,title='Monthly revenue trend',color_discrete_sequence=['#287f86']))
    with b: chart(px.bar(view.groupby('Year',as_index=False).TotalAmount.sum(),x='Year',y='TotalAmount',title='Revenue by year',color_discrete_sequence=['#4267a9']))
    day=view.groupby('DayName',as_index=False).TotalAmount.sum(); day['DayName']=pd.Categorical(day.DayName,['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'],ordered=True); day=day.sort_values('DayName'); chart(px.bar(day,x='DayName',y='TotalAmount',title='Revenue by day of week',color_discrete_sequence=['#287f86']))
with t2:
    a,b=st.columns(2); top=view.groupby(['StockCode','Description'],as_index=False).agg(Revenue=('TotalAmount','sum'),Units=('Quantity','sum')).nlargest(12,'Revenue')
    with a: chart(px.bar(top.sort_values('Revenue'),x='Revenue',y='Description',orientation='h',title='Top products by revenue',color_discrete_sequence=['#287f86']))
    country=view.groupby('Country',as_index=False).TotalAmount.sum().nlargest(10,'TotalAmount')
    with b: chart(px.bar(country.sort_values('TotalAmount'),x='TotalAmount',y='Country',orientation='h',title='Top countries by revenue',color_discrete_sequence=['#4267a9']))
    st.dataframe(top,use_container_width=True,hide_index=True)
with t3:
    a,b=st.columns(2)
    with a: chart(px.histogram(cust,x='TotalRevenue',nbins=35,title='Customer revenue distribution',color_discrete_sequence=['#287f86']))
    with b:
        c=cust.groupby('Country',as_index=False).CustomerID.nunique().nlargest(10,'CustomerID'); chart(px.bar(c,x='CustomerID',y='Country',orientation='h',title='Customers by country',color_discrete_sequence=['#4267a9']))
    st.dataframe(cust[['CustomerID','Country','TotalOrders','TotalQuantity','TotalRevenue','AverageOrderValue','Recency']].sort_values('TotalRevenue',ascending=False).head(30),use_container_width=True,hide_index=True)
with t4:
    st.markdown('<div class="section">Star schema dimensions</div>',unsafe_allow_html=True); cs=st.columns(4)
    for col,(name,n) in zip(cs,[('Fact sales',len(fact)),('Dim customer',len(cust)),('Dim product',len(prod)),('Dim date',len(d['date']))]):
        with col: metric(name,f'{n:,} rows')
    st.markdown('<div class="note">Fact sales connects to customer, product and date dimensions through CustomerKey, ProductKey and DateKey. Rows without dimension keys are excluded, as in the notebook.</div>',unsafe_allow_html=True)
    op=st.selectbox('OLAP operation',['Roll-up — yearly summary','Drill-down — year and month','Slice — one year','Dice — selected months and threshold','Pivot — year by month'])
    if op.startswith('Roll-up'): result=fact.groupby('Year',as_index=False).agg(TotalRevenue=('TotalAmount','sum'),TotalQuantity=('Quantity','sum'),TotalTransactions=('InvoiceNo','nunique'))
    elif op.startswith('Drill-down'): result=fact.groupby(['Year','Month','MonthName'],as_index=False).agg(TotalRevenue=('TotalAmount','sum'),TotalQuantity=('Quantity','sum'),TotalTransactions=('InvoiceNo','nunique')).sort_values(['Year','Month'])
    elif op.startswith('Slice'):
        yr=st.selectbox('Year',sorted(fact.Year.unique())); s=fact[fact.Year==yr]; result=s.groupby(['Month','MonthName'],as_index=False).agg(TotalRevenue=('TotalAmount','sum'),TotalQuantity=('Quantity','sum'),TotalTransactions=('InvoiceNo','nunique')).sort_values('Month')
    elif op.startswith('Dice'):
        yr=st.selectbox('Year for dice',sorted(fact.Year.unique())); months=st.multiselect('Months',list(range(1,13)),[1,2,3]); threshold=st.number_input('Minimum transaction amount',min_value=0.0,value=100.0); s=fact[(fact.Year==yr)&fact.Month.isin(months)&(fact.TotalAmount>threshold)]; result=s.groupby(['Year','Month'],as_index=False).agg(TotalRevenue=('TotalAmount','sum'),TotalQuantity=('Quantity','sum'),TotalTransactions=('InvoiceNo','nunique'))
    else: result=fact.pivot_table(index='Year',columns='MonthName',values='TotalAmount',aggfunc='sum',fill_value=0).reset_index()
    st.dataframe(result,use_container_width=True,hide_index=True); st.download_button('Download OLAP result CSV',result.to_csv(index=False).encode(),'olap_result.csv','text/csv')
    with st.expander('Preview warehouse tables'):
        table=st.selectbox('Choose table',['Fact sales','Customer dimension','Product dimension','Date dimension']); frame={'Fact sales':fact,'Customer dimension':cust,'Product dimension':prod,'Date dimension':d['date']}[table]; st.dataframe(frame.head(100),use_container_width=True,hide_index=True)
with t5:
    st.markdown('<div class="section">Random Forest churn classification</div>',unsafe_allow_html=True); st.caption('Churn = no purchases in the next 90-day window. The model settings and feature definitions follow your notebook.')
    if d['metrics'] is None: st.warning('Not enough labelled customers or only one target class in this dataset.')
    else:
        a=st.columns(5)
        for col,(name,val) in zip(a,d['metrics'].items()):
            with col: metric(name,f'{val*100:.2f}%' if name!='ROC-AUC' else f'{val:.4f}')
        a,b=st.columns(2)
        with a:
            counts=churn.Churn.map({0:'Active',1:'Churned'}).value_counts().rename_axis('Status').reset_index(name='Customers'); chart(px.pie(counts,names='Status',values='Customers',hole=.55,title='Customer churn distribution'))
        with b:
            imp=pd.DataFrame({'Feature':FEATURES,'Importance':d['model'].feature_importances_}).sort_values('Importance'); chart(px.bar(imp,x='Importance',y='Feature',orientation='h',title='Feature importance',color_discrete_sequence=['#4267a9']))
        st.write(f"Training population: {len(churn):,} customers · Cutoff date: {d['cutoff'].date()} · Reference date: {d['reference'].date()}")
        st.info('The notebook’s saved run reported 64.39% accuracy, 57.14% precision, 68.97% recall, 62.50% F1 and 0.7089 ROC-AUC. Live results are recalculated from the uploaded dataset and may differ.')
        st.dataframe(churn[['CustomerID']+FEATURES+['FutureOrders','Churn']].head(30),use_container_width=True,hide_index=True)
with t6:
    a=st.columns(4)
    for col,(name,val) in zip(a,[('Original rows',d['original']),('Duplicates removed',d['duplicates']),('Cancelled invoices',d['cancelled']),('Valid sales rows',d['valid'])]):
        with col: metric(name,f'{val:,}')
    st.markdown('**Missing values in uploaded source**'); st.dataframe(d['raw'].isna().sum().rename('Missing values').reset_index().rename(columns={'index':'Column'}),use_container_width=True,hide_index=True)
    checks=pd.DataFrame({'Validation':['Customer keys unique','Product keys unique','Date keys unique','Fact customer keys valid','Fact product keys valid','Fact date keys valid'],'Passed':[cust.CustomerKey.is_unique,prod.ProductKey.is_unique,d['date'].DateKey.is_unique,fact.CustomerKey.isin(cust.CustomerKey).all(),fact.ProductKey.isin(prod.ProductKey).all(),fact.DateKey.isin(d['date'].DateKey).all()]}); st.dataframe(checks,use_container_width=True,hide_index=True)
    for name,frame in [('fact_sales.csv',fact),('dim_customer.csv',cust),('dim_product.csv',prod),('dim_date.csv',d['date'])]: st.download_button(f'Download {name}',frame.to_csv(index=False).encode(),name,'text/csv')
st.markdown('---'); st.caption('RetailIQ · Online Retail Data Mining & Data Warehouse Analysis')
