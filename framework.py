import pandas as pd
import numpy as np
import operator
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics import pairwise_distances
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.model_selection import KFold
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import PowerTransformer
style.use('seaborn-bright')
pd.options.mode.chained_assignment = None
class DataAnalyticalFramework:
    #constructor
    def __init__(self, df_input=None):  
        if df_input is None:
            pass
        else:
            try:
                self.df_input = pd.read_csv(df_input)
            except Exception as e:
                print(e)

    #returns the initialized dataframe
    def get_df(self):
        try:
            return self.df_input
        except Exception as e:
            print(e)

    #sets new dataframe
    def set_new_df(self, new_df):
        try:
            self.df_input = new_df
        except Exception as e:
            print(e)

    #gets the columns of the dataframe
    def get_column(self, from_int=None, to_int=None):
        try:
            if from_int is None and to_int is None: 
                return list(self.df_input)
            else:
                get_col_arr = list(self.df_input)
                column_arr = []
                while from_int < to_int:
                    column_arr.append(get_col_arr[from_int])
                    from_int += 1
                return column_arr
        except Exception as e:
            print(e)

    #prints the columns of the dataframe
    def print_column(self, from_int=None, to_int=None):
        try:
            if from_int is None and to_int is None:
                counter = 1
                for col in list(self.df_input): 
                    print(str(counter) + " " + col) 
                    counter += 1
            else:
                counter = 1
                print_col_arr = list(self.df_input)
                while from_int < to_int:
                    print(str(counter) + " " + print_col_arr[from_int])
                    from_int += 1
                    counter += 1
        except Exception as e:
            print(e)

    #prints array with counters
    def print_arr(self, inp_df):
        try:
            counter = 0
            while counter < len(inp_df):
                print(str(counter+1) + " " + inp_df[counter])
                counter += 1
        except Exception as e:
            print(e)

    #gets the rows of the dataframe
    def get_row(self, identifier, from_int=None, to_int=None):
        try:
            if from_int is None and to_int is None: 
                return self.df_input[identifier]
            else:
                get_row_arr = self.df_input[identifier]
                row_arr = []
                while from_int < to_int:
                    row_arr.append(get_row_arr[from_int])
                    from_int += 1
                return row_arr
        except Exception as e:
            print(e)

    #prints the rows of the dataframe
    def print_row(self, identifier, from_int=None, to_int=None):
        try:
            if from_int is None and to_int is None:
                counter = 1
                for col in self.df_input[identifier]: 
                    print(str(counter) + " " + col) 
                    counter += 1
            else:
                counter = 1
                arr = self.df_input[identifier]
                while from_int < to_int:
                    print(str(counter) + " " + arr[from_int])
                    from_int += 1
                    counter += 1
        except Exception as e:
            print(e)

    #locates and returns rows that contains what is in the conditional input?
    def locate(self, column, cond_inp):
        try:
            return self.df_input.loc[self.df_input[column] == cond_inp]
        except Exception as e:
            print(e)

    #groups frame with a specific identifier
    def group_frame_by(self, identifier, df_input=None):
        try:
            if df_input is None:
                return self.df_input.loc[identifier].sum()
            else:
                return df_input.loc[identifier].sum()
        except Exception as e:
            print(e)

    #groups frame from and to specific column index        
    def group_frame_from(self, identifier, from_int, to_int, df_input=None):
        try:
            if df_input is None:
                first_df = self.df_input.groupby([identifier],as_index=False)[self.df_input.columns[from_int:to_int]].sum()
                return first_df
            else:
                first_df = df_input.groupby([identifier],as_index=False)[df_input.columns[from_int:to_int]].sum()
                return first_df
        except Exception as e:
            print(e)

    #groups frame from and to specific index with the exception of the given last parts       
    def group_frame_except(self, identifier, from_int, to_int, df_input=None):
        try:
            if df_input is None:
                first_df = self.df_input.groupby([identifier],as_index=False)[self.df_input.columns[from_int:to_int]].sum()
                second_df = self.df_input.iloc[: , to_int:]
                return first_df.join(second_df) 
            else:
                first_df = df_input.groupby([identifier])[df_input.columns[from_int:to_int]].sum()
                second_df = df_input.iloc[: , to_int:]
                return first_df.join(second_df) 
        except Exception as e:
            print(e)

    #groups frame with a specific identifier
    def extract_row(self, column, identifier, df_input=None):
        try:
            if df_input is None:
                return self.df_input.loc[self.df_input[column] == identifier]
            else:
                return df_input.loc[df_input[column] == identifier]  
        except Exception as e:
            print(e)

    #get line equation
    def line_eq(self, independent, dependent):
        try:
            m = self.get_slope(independent, dependent)
            b = self.get_intercept(independent, dependent)
            lin_equation = "y = " + str(m) + "x "
            if(b < 0):
                lin_equation += "+ (" + str(m) + ")"
            else:
                lin_equation += "+ " + str(b)
            
            return lin_equation
        except Exception as e:
            print(e)

    #get polynomial equation
    def poly_eq(self, independent, dependent):
        try:
            x = self.df_input[[independent]]
            y = self.df_input[[dependent]]

            poly = PolynomialFeatures(degree = 2)
            x_poly = poly.fit_transform(x)  
            print(x_poly)    

            model = LinearRegression()
            model.fit(x_poly, y)
            coef_arr = model.coef_
            intercept_arr = model.intercept_
            
            poly_equation = "y = " + str(round(intercept_arr[0], 4))
            if(coef_arr[0][0] < 0):
                poly_equation += " + (" + str(round(coef_arr[0][0], 4)) + "x\xB0" + ")"
            else:
                poly_equation += " + " + str(round(coef_arr[0][0], 4)) + "x\xB0"
            
            if(coef_arr[0][1] < 0):
                poly_equation += " + (" + str(round(coef_arr[0][1], 4)) + "x" + ")"
            else:
                poly_equation += " + " + str(round(coef_arr[0][2], 4)) + "x"
            
            if(coef_arr[0][2] < 0):
                poly_equation += " + (" + str(round(coef_arr[0][2], 4)) + "x\xB2" + ")"
            else:
                poly_equation += " + " + str(round(coef_arr[0][2], 4)) + "x\xB2"

            
            return  poly_equation
        except Exception as e:
            print(e)
    
    #get intercept of polynomial regression
    def get_poly_intercept(self, independent, dependent):
        try:
            x = self.df_input[[independent]]
            y = self.df_input[[dependent]]

            poly = PolynomialFeatures(degree = 2)
            x_poly = poly.fit_transform(x)  
            print(x_poly)    

            model = LinearRegression()
            model.fit(x_poly, y)
            intercept_arr = model.intercept_
            return round(intercept_arr[0], 4)
        except Exception as e:
            print(e)
    
    #get coefficients of polynomial regression
    def get_poly_coeff(self, independent, dependent):
        try:
            x = self.df_input[[independent]]
            y = self.df_input[[dependent]]

            poly = PolynomialFeatures(degree = 2)
            x_poly = poly.fit_transform(x)  
            print(x_poly)    

            model = LinearRegression()
            model.fit(x_poly, y)
            return model.coef_
        except Exception as e:
            print(e)

    #get the slope of the line
    def get_slope(self, independent, dependent, second_indep=None):
        try:
            if second_indep is None:
                x = self.df_input[[independent]]
                y = self.df_input[[dependent]]

                lm = LinearRegression()
                lm.fit(x, y)
                m = lm.coef_ 
                return round(m[0][0], 4)
            else:
                x = self.df_input[[independent, second_indep]]
                y = self.df_input[[dependent]]

                lm = LinearRegression()
                lm.fit(x, y)
                m = lm.coef_ 
                return m
        except Exception as e:
            print(e)
    
    #get the y-intercept of the line
    def get_intercept(self, independent, dependent):
        try:
            x = self.df_input[[independent]]
            y = self.df_input[[dependent]]

            lm = LinearRegression()
            lm.fit(x, y)
            b = lm.intercept_
            return round(b[0], 4)
        except Exception as e:
            print(e)

    #get the coefficient of determination (r^2) of the line
    def get_coeff_det(self, independent, dependent, second_indep=None):
        try:
            if second_indep is None:
                x = self.df_input[[independent]]
                y = self.df_input[[dependent]]

                lm = LinearRegression()
                lm.fit(x, y)
                y_pred = lm.predict(x)
                r_score = r2_score(y, y_pred)
                return round(r_score, 4)
            else:
                x = self.df_input[[independent, second_indep]]
                y = self.df_input[[dependent]]

                x = sm.add_constant(x)
                model = sm.OLS(y, x).fit()
                return round(model.rsquared, 4)
        except Exception as e:
            print(e)

    #shows the scatter plot
    def scatter_plot(self, independent, dependent, second_indep=None):
        try:
            if second_indep is None:
                x = self.df_input[independent]
                y = self.df_input[dependent]

                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.scatter(x, y)
                ax.set_xlabel(independent)
                ax.set_ylabel(dependent)
                ax.axis('tight')
                plt.title("Scatter Plot of " + dependent + " and " + independent)
                plt.show()
            else:
                x = self.df_input[[independent]]
                y = self.df_input[[dependent]]
                z = self.df_input[[second_indep]]

                # plot the results
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(x, y, z)
                ax.set_xlabel(independent)
                ax.set_ylabel("Number of cases of " + dependent)
                ax.set_zlabel(second_indep)
                ax.axis('tight')
                plt.title("Scatter Plot of " + dependent + ", " + independent + " and " + second_indep)
                plt.show()
        except Exception as e:
            print(e)

    #shows the linear regression
    def linear_regression(self, independent, dependent):
        try:
            if isinstance(independent, str) and isinstance(dependent, str):
                x = self.df_input[[independent]]
                y = self.df_input[[dependent]]
            elif isinstance(independent, pd.DataFrame) and isinstance(dependent, pd.DataFrame):
                x = independent
                y = dependent

            lm = LinearRegression()
            model = lm.fit(x, y)
            x_new = np.linspace(self.df_input[independent].min() - 5, self.df_input[independent].max() + 5, 50)
            y_new = model.predict(x_new[:, np.newaxis])

            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(x_new, y_new, color = 'blue', label=self.line_eq(independent, dependent))
            ax.legend(fontsize=9)
            ax.scatter(x, y)
            ax.set_xlabel(independent)
            ax.set_ylabel(dependent)
            ax.axis('tight')
            plt.title("Linear Regression of " + independent + " and " + dependent)
            plt.show()
                
        except Exception as e:
            print(e)

    #shows the computations for linear regression
    def linear_reg_summary(self, independent, dependent, second_indep=None):
        try:
            if second_indep is None:
                if isinstance(independent, str) and isinstance(dependent, str):
                    x = self.df_input[[independent]]
                    y = self.df_input[[dependent]]
                elif isinstance(independent, pd.DataFrame) and isinstance(dependent, pd.DataFrame):
                    x = independent
                    y = dependent
                
                x = sm.add_constant(x)
                
                model = sm.OLS(y, x).fit()
                print(model.summary())
            else:
                if isinstance(independent, str) and isinstance(dependent, str) and isinstance(second_indep, str):
                    x = self.df_input[[independent, second_indep]]
                    y = self.df_input[[dependent]]
                elif isinstance(independent, pd.DataFrame) and isinstance(dependent, pd.DataFrame) and isinstance(second_indep, pd.DataFrame):
                    x = independent
                    y = dependent

                x = sm.add_constant(x)
                model = sm.OLS(y, x).fit()
                print(model.summary())
        except Exception as e:
            print(e)

    #shows polynomial regression
    def polynomial_reg(self, independent, dependent):
        try:
            if isinstance(independent, str) and isinstance(dependent, str):
                x = self.df_input[independent]
                y = self.df_input[[dependent]]
            elif isinstance(independent, pd.DataFrame) and isinstance(dependent, pd.DataFrame):
                x = independent
                y = dependent

            x = x[:, np.newaxis]
            y = y[: np.newaxis]

            poly = PolynomialFeatures(degree = 2)
            x_poly = poly.fit_transform(x)  

            model = LinearRegression()
            model.fit(x_poly, y)
            y_poly_pred = model.predict(x_poly)

            plt.scatter(x, y, color = 'red')
            sort_axis = operator.itemgetter(0)
            sorted_zip = sorted(zip(x,y_poly_pred), key=sort_axis)
            x, y_poly_pred = zip(*sorted_zip)
            plt.plot(x, y_poly_pred, color='blue', label=self.poly_eq(independent, dependent))
            plt.legend(fontsize=9)

            plt.title("Polynomial Regression of " + independent + " and " + dependent)
            plt.show()
                
        except Exception as e:
            print(e)

    #shows the computations for polynomial regression
    def polynomial_reg_summary(self, independent, dependent):
        try:
            result = smf.ols(formula=str(dependent) + ' ~ ' + str(independent) + '+ I(' + str(independent) + '**2)', data=self.df_input).fit()
            print(result.summary())
        except Exception as e:
            print(e)

    #returns table summary per category of disease based on simple linear regression
    def linear_reg_table(self, title, independent, dependent, second_indep=None):
        try:
            if second_indep is None:
                coeff = []
                coeff_det = []
                total_occ = []

                for step in dependent:
                    if("†" in step):
                        step = step.replace("†,", "† ,")
                    elif("â€" in step):
                        step = step.replace("â€,", "â€ ,")
                    coeff.append(self.get_slope(independent, step))
                    coeff_det.append(self.get_coeff_det(independent, step))
                    total_occ.append(self.df_input[step].sum())

                table_content =  {title: dependent, "Number of Cases": total_occ, "Coefficient": coeff, "Coefficient of Determination (R^2)": coeff_det}
                table_df = pd.DataFrame(table_content)
                return table_df
            else:
                first_coeff = []
                second_coeff = []
                coeff_det = []
                total_occ = []

                for step in dependent:
                    if("†" in step):
                        step = step.replace("†,", "† ,")
                    elif("â€" in step):
                        step = step.replace("â€,", "â€ ,")
                    m = self.get_slope(independent, step, second_indep)
                    first_coeff.append(round(m[0][0], 4))
                    second_coeff.append(round(m[0][1], 4))
                    coeff_det.append(self.get_coeff_det(independent, step, second_indep))
                    total_occ.append(self.df_input[step].sum())

                table_content =  {title: dependent, "Number of Cases": total_occ, "Coefficient of " + independent: first_coeff, "Coefficient of " + second_indep: second_coeff, "Coefficient of Determination (R^2)": coeff_det}
                table_df = pd.DataFrame(table_content)
                return table_df
        except Exception as e:
            print(e)
    
    #shows table summary per category of disease based on multiple linear regression

    def nb_feature_select(self,estimator, n_features, X, y,cv=5):
        try:

            selector = RFECV(estimator, step=1,cv=cv)
            selector = selector.fit(X,y)
            support = selector.support_
            selected = []
            for a, s in zip(X.columns, support):
                if(s):
                    selected.append(a)
                    print(a)
            return selected
        except Exception as e:
            print(e)

    def naive_bayes(self,X_columns, y_column, cv_kfold=10, class_bins=0, nb_type='complement', n_features=0):
        try:
            
            X = self.df_input[X_columns]
            y = self.df_input[[y_column]]

            scaler = MinMaxScaler()
            for col in X.columns:
                X[col] = scaler.fit_transform(X[[col]].astype(float))
            
            if(class_bins!=0):
                est = KBinsDiscretizer(n_bins=class_bins, encode='ordinal', strategy='kmeans')
                y[[y.columns[0]]] = est.fit_transform(y[[y.columns[0]]])
                # print(est.bin_edges_[0])
            
            if(n_features!=0):
                X = X[self.nb_feature_select(LogisticRegression(solver='lbfgs', multi_class='auto'), n_features, X, y, cv=10)]
                
            X = X.values.tolist()

            y = y[y.columns[0]].values.tolist()

            if(nb_type == 'complement'):
                nb = ComplementNB()
            elif(nb_type == 'gaussian'):
                nb = GaussianNB()
            
            kf = KFold(n_splits=cv_kfold)
            # X = pd.DataFrame(X)
            y_true_values,y_pred_values = [], []


            for train_index, test_index in kf.split(X,y):

                X_test = [X[ii] for ii in test_index]
                X_train = [X[ii] for ii in train_index]
                y_test = [y[ii] for ii in test_index]
                y_train = [y[ii] for ii in train_index]

                y_true_values = np.append(y_true_values, y_test)

                nb.fit(X_train,y_train)
                y_pred =nb.predict(X_test)
                y_pred_values = np.append(y_pred_values, y_pred)
            

            return y_true_values, y_pred_values, accuracy_score(y_true_values,y_pred_values)
            


        except Exception as e:
            print(e)

    def naive_bayes_cm(self,X_columns, y_column,cv_kfold=10, class_bins=0, nb_type='complement', n_features=0):
        try:

            y_true, y_pred, accuracy = self.naive_bayes(X_columns, y_column, cv_kfold=cv_kfold, class_bins=class_bins, nb_type=nb_type, n_features=n_features)
            
            cm = confusion_matrix(y_true, y_pred)
            cm_norm = (cm.astype('float') / cm.sum(axis=1)[:, np.newaxis])*100
                
            fig, ax = plt.subplots()
            im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            ax.figure.colorbar(im, ax=ax)

            ax.set(xticks=np.arange(cm.shape[1]),yticks=np.arange(cm.shape[0]),ylabel='True Label',xlabel='Predicted Label')

            thresh = cm.max() / 2.
            for x in range(cm_norm.shape[0]):
                for y in range(cm_norm.shape[1]):
                    if(x==y):
                        ax.text(y,x,f"{cm[x,y]}({cm_norm[x,y]:.2f}%)",ha="center", va="center", fontsize=12, color="white" if cm[x, y] > thresh else "black")
                    else:
                        ax.text(y,x,f"{cm[x,y]}({cm_norm[x,y]:.2f}%)",ha="center", va="center", color="white" if cm[x, y] > thresh else "black")
            
            ax.annotate("Accuracy: "+ str(accuracy),xy=(10, 10), xycoords='figure pixels')
            plt.title("Naive Bayes Confusion Matrix ("+y_column+")")
            plt.show()
        except Exception as e:
                print(e)

    def km_feature_select(self,X_columns, y_column, k, min_sil_score=0.5):          
        try:
            X = self.df_input[X_columns]
            y = self.df_input[[y_column]]
            features = pd.concat([X,y],axis=1)
            features_selected = [y_column]
            sil_score = 1
            ctr = len(X_columns)
            while(sil_score>min_sil_score and ctr>0):
                temp_selected = ""
                temp_coef = 0
                for col in X_columns:
                    temp_feat_sel = np.append(features_selected,col)
                    kmeans_model = KMeans(n_clusters=k,random_state=0).fit(features[temp_feat_sel])
                    labels = kmeans_model.labels_
                    sil_coef = metrics.silhouette_score(features[temp_feat_sel], labels, metric='euclidean')
                    if(col not in features_selected and sil_coef>temp_coef and sil_coef>min_sil_score):
                        temp_coef = sil_coef
                        temp_selected = col
                        print(temp_coef)
                sil_score = temp_coef
                if(sil_score>min_sil_score):
                    features_selected = np.append(features_selected,temp_selected)
                ctr -= 1
            return features_selected

        except Exception as e:
                    print(e)
    
    def k_means(self,X_columns, y_column, k, min_sil_score=0, k_finder=0): ##change k_finder name

        try:
            X = self.df_input[X_columns]
            y = self.df_input[[y_column]]
            features = pd.concat([X,y],axis=1)
            scaler = MinMaxScaler()
            for col in features.columns:
                features[col] = scaler.fit_transform(features[[col]].astype(float))
            if(min_sil_score!=0):
                sel_feat = self.km_feature_select(X_columns, y_column, 5, min_sil_score=min_sil_score)
                features = features[sel_feat]

            if(k_finder==0):
                kmeans_model = KMeans(n_clusters=k).fit(features)
                labels = kmeans_model.labels_
                sil_coef = metrics.silhouette_score(features, labels, metric='euclidean')
                centroids = kmeans_model.cluster_centers_          
            
            elif(k_finder==1):
                sil_coef = 0
                centroids = [[]]
                labels = []
                for k in range(2,k):
                    kmeans_model = KMeans(n_clusters=k).fit(features)
                    labels_temp = kmeans_model.labels_
                    sil_coef_temp = metrics.silhouette_score(features, labels_temp, metric='euclidean')
                    centroids_temp = kmeans_model.cluster_centers_
                    if(sil_coef_temp>sil_coef):
                        sil_coef = sil_coef_temp
                        centroids = centroids_temp
                        labels = labels_temp          
            
            return centroids,sil_coef,features,labels

        except Exception as e:
            print(e)

    def k_means_cc(self,X_columns, y_column ,k , min_sil_score=0, k_finder=0):
        try:
            centroids, sil_coef,features,labels = self.k_means(X_columns, y_column, k, min_sil_score=min_sil_score, k_finder = k_finder)
            
            df = pd.DataFrame(data=labels)

            df = pd.concat([features,df], axis=1)
            print(df)
            print(sil_coef)
            print(centroids)

            fig, ax = plt.subplots()
            for k in range(centroids.shape[0]):
                plt.plot(range(centroids.shape[1]),centroids[k], label="Cluster "+str(k))
            plt.xticks(range(centroids.shape[1]),features.columns)
            plt.setp( ax.xaxis.get_majorticklabels(), rotation=-15, ha="left" )
            plt.xlabel("Features")
            plt.legend(bbox_to_anchor=(1.05, 1), loc=0, borderaxespad=0.)
            plt.title("Centroid Chart ("+y_column+")")
            plt.tight_layout()
            plt.show()

        except Exception as e:
                print(e)