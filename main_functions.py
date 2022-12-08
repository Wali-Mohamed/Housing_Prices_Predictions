import pandas as pd
import numpy as np
from scipy.stats import multivariate_normal as mvn

def wrangle(path):
    # read dataframe
    df= pd.read_csv(path)
    # replace 'None' with None so it will show when checking for missing values
    df['bathrooms']=df['bathrooms'].replace('None',None).astype(float)
    df['sqrt_ft']=df['sqrt_ft'].replace('None',None).astype(float)
    df['garage']=df['garage'].replace('None',None).astype(float)
    df['fireplaces']=df['fireplaces'].replace(' ',None).astype(float)
    df['HOA']= df['HOA'].str.replace(',','').replace('None', None).astype(float)
    # drop high cardinality and unnecessary columns
    df.drop(columns=['kitchen_features', 'floor_covering', 'year_built'], inplace=True)
    # round the decimal values in number of bathroom
    df.bathrooms=df.bathrooms.replace({2.5:3, 3.5:4, 4.5:5})
    # round the decimal values in number of garages
    df.garage=df.garage.replace({2.5:3, 3.5:4, 4.5:5})
    #drop high colinearity
    df.drop(columns=['zipcode', 'bedrooms', 'bathrooms'], inplace=True)
    # drop leaky and low correlation columns
    df.drop(['lot_acres', 'taxes', 'MLS'], axis=1, inplace=True)
    # fill na with median since HOA column is skewed
    df['HOA']=df['HOA'].fillna(df['HOA'].median())
    # remove top 10%  from the top and 5% from the bottom of the data because outliers can not be generalized
    low, high= df['sold_price'].quantile([0.05, 0.90])
    mask_no_outlier=df['sold_price'].between(low, high)
    df=df[mask_no_outlier]
    # drop NaN
    df.dropna(inplace=True)
   
    

    
    
    return df
def accuracy(y, y_hat):
    return np.mean(y==y_hat)

def StandardScaler(X):
    result=(X-np.mean(X))/np.std(X)
    return result

def MinMaxScaler(X):
    result=((X-X.min())/(X.max()-X.min()))
    return result
def R2score(y, y_hat):
    y_bar=y.mean()
    ssr=np.sum((y-y_hat)**2)
    sst=np.sum((y-y_bar)**2)
    score=1-(ssr/sst)
    return score

def mae(y, y_hat):
    total=np.sum(abs(y-y_hat))
    return (total/len(y))

class KNNclassifier():
    def fit(self, X, y):
        self.X=X
        self.y=y
    def predict(self, X, k , epsilon=1e-3):
        N = len(X)   # number of rows
        y_hat = np.zeros(N)
        for i in range(N):
              dist2 = np.sum((self.X-X[i])**2, axis=1) 
              idxt = np.argsort(dist2)[:k]
              gamma_k =1/np.sqrt(dist2[idxt]+epsilon)
              y_hat[i] =np.bincount(self.y[idxt], weights=gamma_k).argmax()
        return y_hat
    
class Gauss():
    def fit(self, X, y, epsilon=1e-3):
        self.likelihoods=dict()
        self.priors = dict()
        self.K = set(y.astype(int))
    
        for k in self.K:
            
            X_k = X[y==k,:]

            N_k, D = X_k.shape

            Mu_k = X_k.mean(axis=0)
            
            self.likelihoods[k]={'mean':Mu_k, 'cov':(1/(N_k-1))*np.matmul((X_k-Mu_k).T,X_k-Mu_k)  + epsilon*np.identity(D)}
            self.priors[k] = len(X_k)/len(X)
            
    def predict(self, X):
        N, D = X.shape
        P_hat =np.zeros((N, len(self.K)))
        
        
        
        for k, l in self.likelihoods.items():
            
            # Apply Bayes Theorem
            P_hat[:,k]=mvn.logpdf(X, l['mean'],l['cov'])+np.log(self.priors[k])
        
        return P_hat.argmax(axis=1) 
