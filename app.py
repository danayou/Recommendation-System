import pandas as pd
import numpy as np
from flask import Flask, render_template, request
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('/Users/danayou/Desktop/DS-Case-Study/data/customer.csv')
df = df.head(15000)

df = df[~df['Sales_Order_Quantity'].isin([0.0])]
df = df.drop(df[df['Cancelled_Flag'] == 'False'].index)
df['Purchase_or_Return'] = df['Sales_Order_Quantity'].apply(lambda x: 'Purchase' if x > 0 else 'Return')
df['Location'] = df['Customer_ShipTo_Addr3'] + ', ' + df['Customer_ShipTo_Addr4']
df['Single_Margin'] = abs(df['Sales_Order_Extended_Margin_Amount'] / df['Sales_Order_Quantity'])

df['DATE_KEY'] = pd.to_datetime(df['DATE_KEY'], format='%m/%d/%Y')
df['Join_Year'] = df['DATE_KEY'].dt.year
customer_join_year = df.groupby('CUSTOMER_ID')['Join_Year'].min().reset_index()
customer_join_year.rename(columns={'Join_Year': 'Year_Joined'}, inplace=True)

returns_df = df[df['Sales_Order_Quantity'] < 0]
returned_df = returns_df.groupby(['CUSTOMER_ID', 'Product_Name']).agg({
    'Sales_Order_Quantity': 'sum',     
}).reset_index()
returned_df.rename(columns={'Sales_Order_Quantity': 'Returns'}, inplace=True)

result_df = df.groupby(['CUSTOMER_ID', 'Product_Name', 'Purchase_or_Return'], as_index=False).agg({
    'Sales_Order_Quantity': 'sum',
    'Single_Margin': 'mean'}
)
result_df.rename(columns={'Sales_Order_Quantity': 'Total_Quantity'}, inplace=True)

alpha = 0.6  
beta = 0.4   
penalty_factor = 0.3

result_df['Weighted_Rating'] = np.where(
    result_df['Purchase_or_Return'] == 'Return',
    (alpha * result_df['Total_Quantity'] + beta * result_df['Single_Margin']) * penalty_factor,
    alpha * result_df['Total_Quantity'] + beta * result_df['Single_Margin']
)

customer_encoder = LabelEncoder()
product_encoder = LabelEncoder()

result_df['Customer_Index'] = customer_encoder.fit_transform(result_df['CUSTOMER_ID'])
result_df['Product_Index'] = product_encoder.fit_transform(result_df['Product_Name'])

pivot_df = result_df.pivot_table(
    index='CUSTOMER_ID',      
    columns='Product_Name',     
    values='Weighted_Rating',   
    fill_value=0               
)

csr = csr_matrix(pivot_df.values)

n_components = 500  
svd = TruncatedSVD(n_components=n_components, random_state=42)

user_features = svd.fit_transform(csr)
item_features = svd.components_

predicted_ratings = np.dot(user_features, item_features)
predicted_df = pd.DataFrame(predicted_ratings, index=pivot_df.index, columns=pivot_df.columns)

app = Flask(__name__)

#homepage
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

#recommendation results
@app.route('/search', methods=['POST'])
def get_recommendations():
    cid = request.form.get('search_query')
    user_row = predicted_df.loc[cid]
    unrated_products = pivot_df.loc[cid] == 0
    recommended_products = user_row[unrated_products].sort_values(ascending=False)
    top_10 = recommended_products.head(10)

    return render_template('results.html', search_query=cid, results=top_10.index.tolist())

#purchase history results
@app.route('/history', methods=['POST'])
def user_history():
    cid = request.form.get('customer_id')
    product_names = df[df['CUSTOMER_ID'] == cid]['Product_Name'].tolist()

    return render_template('history.html', customer_id=cid, history=product_names)

#return history results
@app.route('/returns', methods=['POST'])
def return_history():
    cid = request.form.get('customer_id')
    customer_returns = returned_df[returned_df['CUSTOMER_ID'] == cid]['Product_Name'].tolist()

    return render_template('returns.html', customer_id=cid, returns=customer_returns)

#location results
@app.route('/location', methods=['POST'])
def location():
    cid = request.form.get('customer_id')
    loc = df[df['CUSTOMER_ID'] == cid]['Location'][0]

    return render_template('location.html', customer_id=cid, location=loc)

#year results
@app.route('/year', methods=['POST'])
def year():
    cid = request.form.get('customer_id')
    year = customer_join_year[customer_join_year['CUSTOMER_ID'] == cid]['Year_Joined'].values[0]

    return render_template('year.html', customer_id=cid, year=year)

if __name__ == '__main__':
    app.run(port=8000, debug=True)