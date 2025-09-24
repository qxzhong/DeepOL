import numpy as np
import pandas as pd
import math
import torch
import torch.optim as optim
from torch import nn, Tensor
from torchtuples import Model
import copy 
import torchtuples as tt
import scipy.stats as stats
import statsmodels.formula.api as smf
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper
from sklearn.model_selection import train_test_split,KFold
from sklearn_quantile import RandomForestQuantileRegressor,SampleRandomForestQuantileRegressor
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.ensemble import RandomForestClassifier


# Define the quantile loss function
def quantile_loss(y_true, y_pred, tau):
    u = y_true - y_pred
    return torch.mean(u * (tau - (u < 0).float()))

# Define the neural network model
class QuantileNN(nn.Module):
    def __init__(self, input_size, hidden_layers, hidden_nodes):
        super(QuantileNN, self).__init__()
        layers = []
        layers.append(nn.Linear(input_size, hidden_nodes))
        layers.append(nn.ReLU())
        
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_nodes, hidden_nodes))
            layers.append(nn.ReLU())
        
        layers.append(nn.Linear(hidden_nodes, 1))  # Output layer
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class rHQET():
    def __init__(self,df,y_colname,tre_colname):
        super(rHQET,self).__init__()
        self.y_colname,self.tre_colname = y_colname,tre_colname
        x_df = df.loc[:,~df.columns.isin([y_colname,tre_colname])]
        self.x_mean,self.x_std = x_df.mean(axis=0),x_df.std(axis=0)
        self.df = (x_df-self.x_mean)/self.x_std
        self.df[y_colname],self.df[tre_colname] = df[y_colname],df[tre_colname]
        self.n = self.df.shape[0]
        
    def fit(self,tau=0.5,alpha_level=0.05,k_fold=5,denOrder2=True,
            logitModel="linear",qrModel="linear",DML=2,
            n_estimators=100,min_samples_split=3,min_samples_leaf=10,
            lr=None,nodes=[3,64],batch_size=256,epochs=512,random_state=1):
        df = self.df
        self.tau = tau
        df.insert(0,"Intercept",[1.0 for _ in range(df.shape[0])])
        y_colname,tre_colname = self.y_colname,self.tre_colname
        y_df,tre_df = df[y_colname],df[tre_colname]
        x_df = df.loc[:,~df.columns.isin([y_colname,tre_colname])]

        kf = KFold(n_splits=k_fold, shuffle=True, random_state=random_state)

        
        params_list = []
        u_matrix = []
        v_matrix = []
        self.yx_dml2 = []
        self.yx_deep = []
        
        u_matrix_1 = []
        v_matrix_1 = []
        
        for train_index, test_index in kf.split(x_df):
            x_train,y_train,tre_train = x_df.iloc[train_index],y_df.iloc[train_index],tre_df.iloc[train_index]
            x_test,y_test,tre_test = x_df.iloc[test_index],y_df.iloc[test_index],tre_df.iloc[test_index]

            y_pred_list = []
            for t_uniq in [0,1]: # random forest to quantile regression
                tre_index = (tre_train==t_uniq)
                x_train_new,y_train_new = x_train[tre_index],y_train[tre_index]
                h = min([x_train_new.shape[0]**(-1/6),tau*(1-tau)/2])
                if denOrder2:
                    quantiles = [tau+h,tau,tau-h]
                else:
                    quantiles = [tau+2*h,tau+h,tau,tau-h,tau-2*h]
                for i, quantile in enumerate(quantiles):
                    if qrModel=="linear":
                        df_train_new = x_train_new.copy()
                        x_vars = ' + '.join(list(x_train_new.columns)[0:])
                        df_train_new[y_colname] = y_train_new
                        qr_model = smf.quantreg(f"{y_colname}~{x_vars}-1",data=df_train_new)
                        qr_model1 = qr_model.fit(q=quantile)
                        qr_pred = qr_model1.predict(x_test)
                        y_pred_list.append(qr_pred.values.reshape(len(qr_pred),1))
                    else:
                        if x_train_new.shape[0]>10000:
                            qr_model = SampleRandomForestQuantileRegressor(q=quantile,n_estimators=n_estimators,
                                                                           min_samples_split=min_samples_split,
                                                                           min_samples_leaf=min_samples_leaf,
                                                                           random_state=random_state)                  
                        else:
                            qr_model = RandomForestQuantileRegressor(q=quantile,n_estimators=n_estimators,
                                                                     min_samples_split=min_samples_split,
                                                                     min_samples_leaf=min_samples_leaf,random_state=random_state)
                        qr_model.fit(x_train_new,y_train_new)
                        y_pred = qr_model.predict(x_test)
                        y_pred_list.append(y_pred.reshape(len(y_pred),1))
                        

            if denOrder2:
                deltaEta0 = abs(y_pred_list[2]-y_pred_list[0])
                deltaEta1 = abs(y_pred_list[5]-y_pred_list[3])
                eta0,eta1 = y_pred_list[1],y_pred_list[4]
            else:
                deltaEta0 = abs(3*abs(y_pred_list[3]-y_pred_list[1])/2-abs(y_pred_list[4]-y_pred_list[0])/6)
                deltaEta1 = abs(3*abs(y_pred_list[8]-y_pred_list[6])/2-abs(y_pred_list[9]-y_pred_list[5])/6)
                eta0,eta1 = y_pred_list[2],y_pred_list[7]
                

            # propensity score
            if logitModel=="linear":
                lg_model = LogisticRegression(solver='liblinear', random_state=0)
                lg_model.fit(x_train,tre_train)
                propScore = lg_model.predict_proba(x_test)[:,1].reshape(x_test.shape[0],1)
            elif logitModel=="deep":
                propScore = logitTrain(x_train,tre_train,x_test,lr=lr,nodes=nodes,
                                       batch_size=batch_size,epochs=epochs)
            else:
                rf = RandomForestClassifier(n_estimators=10, random_state=25)
                rf.fit(x_train, tre_train)
                probs = rf.predict_proba(x_test)[:,1]
                propScore=probs.reshape((len(probs),1))
                            # get function $\omega(x)$
            A,B = propScore*deltaEta0,(1.0-propScore)*deltaEta1
            wtFtn = A/(A+B)
            wtFtn[A+B==0] = 0.0
            # get $\nu(x)$
            nuFtn = wtFtn*eta1 + (1.0-wtFtn)*eta0

            y_test_new = y_test.values.reshape(nuFtn.shape) - nuFtn
            coef = tre_test.values.reshape(wtFtn.shape)-wtFtn
            df_test_new = coef*x_test

            df_test_new[y_colname] = y_test_new
            x_vars = ' + '.join(list(x_test.columns)[0:])
            qr_linear = smf.quantreg(f"{y_colname}~{x_vars}-1",data=df_test_new)
            params_list.append(qr_linear.fit(q=tau).params)
            
            
            self.yx_dml2.append(df_test_new)
            
            #dataframe for deep learning
            df_deep = x_test.copy()
            df_deep["A_minus_weights"] = coef
            df_deep[y_colname] = y_test_new
            
            self.yx_deep.append(df_deep)
            
            # asymptotic covariance
            deltaEta0[deltaEta0==0.0],deltaEta1[deltaEta1==0.0] = 0.1, 0.1
            density0,density1 =2*h/deltaEta0,2*h/deltaEta1
            coef1,coef0 = propScore*((1.0-wtFtn)**2), (1-propScore)*((wtFtn)**2)
            
            ########################
            self.densities =[density0,density1,coef1,coef0,deltaEta0] 
            p_1 = x_test.shape[1]
            U_k = np.zeros((p_1, p_1))  
            V_k = np.zeros((p_1, p_1)) 
            weight_U = coef1*density1 + coef0*density0
            weight_V = coef1+coef0
            for ii in range(len(coef0)):
                X_i = x_test.iloc[ii].values.reshape(-1, 1)
                U_k += weight_U[ii] * (X_i @ X_i.T)  
                V_k += weight_V[ii] * (X_i @ X_i.T)
            #########################
            
            U_half = (np.sqrt(coef1*density1+coef0*density0)*x_test).to_numpy()
            Sigma_half = (np.sqrt(coef1+coef0)*x_test).to_numpy()
            u_matrix.append(np.dot(U_half.transpose(),U_half)/len(coef0))
            v_matrix.append(tau*(1-tau)*np.dot(Sigma_half.transpose(),Sigma_half)/len(coef0))
            
            
            ########################
            u_matrix_1.append(U_k/len(coef0))
            v_matrix_1.append(V_k/len(coef0))
            self.u_matrix = u_matrix
            self.u_matrix_1 = u_matrix_1
            ########################

        self.df_deep = pd.concat(self.yx_deep)
        del self.df_deep["Intercept"]
       
        df_dml2 = pd.concat(self.yx_dml2)
        qr_linear_dml2 = smf.quantreg(f"{y_colname}~{x_vars}-1",data=df_dml2)
        self.df_params_dml2 = qr_linear_dml2.fit(q=tau).params
        
        
        
        if DML==1:
            df_params = pd.DataFrame(np.array(params_list)).transpose()
            params_mean = df_params.mean(axis=1)
            self.params_mean1=params_mean
        else:
            params_mean = self.df_params_dml2
        
        for i,coef in enumerate(params_mean):
            if i==0:
                s = coef
            else:
                s += -coef*self.x_mean[i-1]/self.x_std[i-1]
                params_mean[i] = coef/self.x_std[i-1]
        params_mean[0] = s
        self.params = pd.Series([i for i in params_mean],index=[i for i in x_test.columns])
        
        # asymptotic covariance
        U = 0
        V = 0
        for k in range(k_fold):
            U += u_matrix[k]
            V += v_matrix[k]
        U,V = U/k_fold,V/k_fold
        U_inv = np.linalg.inv(U)
        cov = np.dot(np.dot(U_inv,V),U_inv)
        self.cov = cov 
        
        lower = []
        upper = []
        stds = []
        
        # confidence interval with leve alpha
        for k in range(x_df.shape[1]):
            if k==0:
                lower.append(params_mean[k] + stats.norm.ppf(alpha_level/2)*np.sqrt(cov[k,k])/np.sqrt(df.shape[0]))
                upper.append(params_mean[k] - stats.norm.ppf(alpha_level/2)*np.sqrt(cov[k,k])/np.sqrt(df.shape[0]))
                stds.append(np.sqrt(cov[k,k])/np.sqrt(df.shape[0]))
            else:
                lower.append(params_mean[k] + 
                             (1/self.x_std[k-1])*stats.norm.ppf(alpha_level/2)*np.sqrt(cov[k,k])/np.sqrt(df.shape[0]))
                upper.append(params_mean[k] - 
                             (1/self.x_std[k-1])*stats.norm.ppf(alpha_level/2)*np.sqrt(cov[k,k])/np.sqrt(df.shape[0]))
                stds.append(np.sqrt(cov[k,k])/np.sqrt(df.shape[0]))
        
        self.stds = stds  
        self.params_df = pd.DataFrame({"coef":[i for i in params_mean],"lower":lower,"upper":upper,"sd":stds},
                                   index=[i for i in x_test.columns])
        
        del df["Intercept"]
        
    def qte_est_intv(self,df_new,alpha_level=0.05):
        self.tau_x_new = self.predict(df_new)
        self.x_new_scale = (df_new[self.x_mean.index]-self.x_mean)/self.x_std
        self.x_new_scale.insert(0,"Intercept",[1.0 for _ in range(self.x_new_scale.shape[0])])
        sd = np.sqrt(np.diagonal((self.x_new_scale.to_numpy() @ self.cov @self.x_new_scale.to_numpy().T)))/np.sqrt(self.n)
        lower = self.tau_x_new-stats.norm.ppf(1-alpha_level/2)*sd
        upper = self.tau_x_new+stats.norm.ppf(1-alpha_level/2)*sd
        return pd.DataFrame({"est":self.tau_x_new,"lower":lower,"upper":upper,"sd":sd})

    # predicted CQTE for linear approach
    def predict(self,df_new):
        s = 0
        for col in self.params.index:
            if (col in df_new.columns):
                s += self.params[col]*df_new[col]
        s += self.params["Intercept"]
        return s
    



     # Training DNN
    def deep_train(self, hidden_layers, hidden_nodes, epochs=1000, lr=0.0001,val_size=0.2):
        tau = self.tau
        df = self.df_deep.copy()
        W = torch.tensor(df["A_minus_weights"].values, dtype=torch.float32).view(-1, 1) #df["A_minus_weights"]
        Y = torch.tensor(df[self.y_colname].values, dtype=torch.float32).view(-1, 1)#df[y_colname]
        X = torch.tensor(df.loc[:,~df.columns.isin([self.y_colname,"A_minus_weights"])].values, dtype=torch.float32)
        

        input_size = X.shape[1]
        model = QuantileNN(input_size, hidden_layers, hidden_nodes)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        # Split data into training and validation sets  
        X_train, X_val, Y_train, Y_val, W_train, W_val = train_test_split(X, Y, W, test_size=val_size, random_state=42)  
        best_val_loss = float('inf')
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()

            # Forward pass
            g_X = model(X_train)
            loss = quantile_loss(Y_train - W_train * g_X, torch.zeros_like(Y_train), tau)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            # Validation  
            model.eval()  
            with torch.no_grad():  
                g_X_val = model(X_val)  
                val_loss = quantile_loss(Y_val - W_val * g_X_val, torch.zeros_like(Y_val), tau)

            if (epoch + 1) % 1000 == 4000:
                print(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')
            # Save the best model 
            if val_loss < best_val_loss:  
                best_val_loss = val_loss  
                patience_counter = 0  
                self.deep_model = copy.deepcopy(model)  
        #self.deep_model1 = model
        return self.deep_model 

    # predicted CQTE for DNN approach
    def nonlinear_predict(self,df_new):
        x_new_scale_ = (df_new[self.x_mean.index]-self.x_mean)/self.x_std
        x_new_scale = torch.tensor(x_new_scale_.values, dtype=torch.float32)
        self.deep_model.eval()  #Set the model to evaluation mode  
        with torch.no_grad():  
            prediction = self.deep_model(x_new_scale)
        self.prediction_ = prediction
        return prediction.numpy().reshape(df_new.shape[0])



        
        
  
    
  

































    
    
    
    
 
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
