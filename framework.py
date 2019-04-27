<<<<<<< HEAD
"""
This program is a framework for generating visualizations such as scatter plot, simple linear regression, multiple linear regression, polynomial regression, K-means clustering and Naive-Bayes classification.
"""

import operator
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from matplotlib import style
from matplotlib import cm
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import ComplementNB
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import r2_score
from sklearn.metrics import r2_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import pairwise_distances
from sklearn.metrics import silhouette_score
from sklearn.model_selection import KFold
from sklearn.feature_selection import RFECV
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import PowerTransformer
pd.options.mode.chained_assignment = None
style.use('seaborn-bright')

class DataAnalyticalFramework:
    """
    This represents the class for the data analytical framework for generating the visualizations.
    """
    version = "1.0"

    def __init__(self, df_input=None):  
        """

        Initializes the use of the class and its functions 
        
        If a dataframe input is specified, it initializes that dataframe as the source of data that will be used throughout the program

        Parameters
        ----------
        df_input : str, optional
            the data frame where the visualizations will be based from
        """

        if df_input is None:
            pass
        else:
            try:
                self.df_input = pd.read_csv(df_input)
            except Exception as e:
                print(e)

    def get_df(self):
        """
        Returns the initialized dataframe

        Returns
        -------
        pandas Dataframe
            initialized dataframe
        """

        try:
            return self.df_input
        except Exception as e:
            print(e)

    def set_new_df(self, new_df):
        """

        Sets a new dataframe

        Parameters
        ----------
        new_df : str, pandas Dataframe
            the dataframe where the visualizations will be based from
        """

        try:
            if(isinstance(new_df, pd.DataFrame)):
                self.df_input = new_df
            if(isinstance(new_df, str)):
                self.df_input = pd.read_csv(new_df)
        except Exception as e:
            print(e)

    def get_column(self, from_int=None, to_int=None):
        """

        Returns the columns of the dataframe

        If a specific range is given, it will return the column names of the dataframe within that specific range

        Parameters
        ----------
        from_int : int, optional
            the index start of the column names to be returned

        to_int : int, optional
            the index end of the column names to be returned

        Returns
        -------
        list
            list of column names
        """

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

    def print_column(self, from_int=None, to_int=None):
        """

        Prints the columns of the dataframe

        If a specific range is given, it will print the column names of the dataframe within that specific range

        Parameters
        ----------
        from_int : int, optional
            the index start of the column names to be printed
        to_int : int, optional
            the index end of the column names to be printed
        """

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

    def print_arr(self, inp_df):
        """

        Prints the contents of the given list with counters from 1 to end of the length of the list

        Parameters
        ----------
        inp_df : list
            the list to be printed with counters
        """

        try:
            counter = 0
            while counter < len(inp_df):
                print(str(counter+1) + " " + inp_df[counter])
                counter += 1
        except Exception as e:
            print(e)

    def get_row(self, identifier, from_int=None, to_int=None):
        """

        Returns the rows of a specified column in the dataframe

        Returns a list that contains the rows of the dataframe within a specific range and identifier

        Parameters
        ----------
        identifier : str
            name of the columns
        from_int : int, optional
            the index start of the row contents to be returned
        to_int : int, optional
            the index end of the row contents to be returned

        Returns
        -------
        pandas Dataframe
            rows of the specified column

        list
            alternatively returns rows of the specified column and within the specified range
        """
        
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

    def print_row(self, identifier, from_int=None, to_int=None):
        """

        Prints the rows of a specified column in the dataframe

        Prints all the rows of the dataframe within a specific range and identifier 

        Parameters
        ----------
        identifier : str
            name of the columns
        from_int : int, optional
            the index start of the row contents to be returned
        to_int : int, optional
            the index end of the row contents to be returned
        """

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

    def locate(self, column, cond_inp):
        """

        Returns a boolean Series based on locating certain rows of a specified column which satisfies a specified condition
        
        Parameters
        ----------
        column : str
            the column of the dataframe where the rows will located at and compared with cond_inp
        cond_inp : str
            the conditional input that will be compared with the contents of the rows of the specified column

        Returns
        -------
        boolean Series
            series containing rows of a specified column which satisfies a specified condition
        """

        try:
            return self.df_input.loc[self.df_input[column] == cond_inp]
        except Exception as e:
            print(e)

    def group_frame_by(self, identifier, df_input=None):
        """

        Returns a Series which is grouped based on a certain identifier

        Parameters
        ----------
        identifier : str
            the identifier which will serve as the basis for grouping the dataframe
        df_input : pandas Dataframe, optional
            custom dataframe to be grouped

        Returns
        -------
        Series
            series containing the grouped rows based on a certain identifier
        """

        try:
            if df_input is None:
                return self.df_input.loc[identifier].sum()
            else:
                return df_input.loc[identifier].sum()
        except Exception as e:
            print(e)
      
    def group_frame_from(self, identifier, from_int, to_int, df_input=None):
        """

        Returns a Series which is grouped based on a certain identifier and a specific column range

        Parameters
        ----------
        identifier : str
            the identifier which will serve as the basis for grouping the dataframe
        from_int : int
            the index start of the column/s to be grouped
        to_int: int
            the index end of the column/s to be grouped
        df_input: pandas Dataframe, optional
            custom dataframe to be grouped

        Returns
        -------
        Series
            series containing the grouped rows based on a certain identifier and a specific column range
        """

        try:
            if df_input is None:
                first_df = self.df_input.groupby([identifier],as_index=False)[self.df_input.columns[from_int:to_int]].sum()
                return first_df
            else:
                first_df = df_input.groupby([identifier],as_index=False)[df_input.columns[from_int:to_int]].sum()
                return first_df
        except Exception as e:
            print(e)
      
    def group_frame_except(self, identifier, from_int, to_int, df_input=None):
        """

        Returns a Series which is grouped based on a certain identifier and a specific column range wherein the outliers of the specified index end are excluded from grouping but will still be part of the Series returned

        Parameters
        ----------
        identifier : str
            the identifier which will serve as the basis for grouping the dataframe
        from_int : int
            the index start of the column/s to be grouped
        to_int : int
            the index end of the column/s to be grouped
        df_input : pandas Dataframe, optional
            custom dataframe to be grouped
        
        Returns
        -------
        Series
            series containing the grouped rows based on a certain identifier and a specific column range with the exception of the outliers
        """

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

    def extract_row(self, column, identifier, df_input=None):
        """

        Returns a boolean Series containing the content of rows based on the specified column which matches a specific identifier
        
        Parameters
        ----------
        column : str
            the column of the dataframe where the rows will extracted at
        identifier : str
            the identifier of the rows to be extracted
        df_input : pandas Dataframe, optional
            custom dataframe to be grouped

        Returns
        -------
        boolean Series
            series containing rows of a specific column which matches a specific identifier
        """

        try:
            if df_input is None:
                return self.df_input.loc[self.df_input[column] == identifier]
            else:
                return df_input.loc[df_input[column] == identifier]  
        except Exception as e:
            print(e)

    def get_poly_intercept(self, independent, dependent):
        """

        Returns the calculated intercept of the polynomial regression
        
        Parameters
        ----------
        independent : str
            the independent(x) variable specified
        dependent : str
            the dependent(y) variable specified

        Returns
        -------
        float
            intercept of the polynomial regression
        """

        try:
            x = self.df_input[[independent]]
            y = self.df_input[[dependent]]

            poly = PolynomialFeatures(degree = 2)
            x_poly = poly.fit_transform(x)   

            model = LinearRegression()
            model.fit(x_poly, y)
            intercept_arr = model.intercept_
            return round(intercept_arr[0], 4)
        except Exception as e:
            print(e)
    
    def get_poly_coeff(self, independent, dependent):
        """

        Returns a list containing the correlation coefficients of the polynomial regression
        
        Parameters
        ----------
        independent : str
            the independent(x) variable specified
        dependent : str
            the dependent(y) variable specified

        Returns
        -------
        list
            list of correlation coefficients of the polynomial regression
        """

        try:
            x = self.df_input[[independent]]
            y = self.df_input[[dependent]]

            poly = PolynomialFeatures(degree = 2)
            x_poly = poly.fit_transform(x)   

            model = LinearRegression()
            model.fit(x_poly, y)
            return model.coef_
        except Exception as e:
            print(e)

    def get_poly_coeff_det(self, independent, dependent):
        """

        Returns the calculated coefficient of determination(R²) of the polynomial regression
        
        Parameters
        ----------
        independent : str
            the independent(x) variable specified
        dependent : str
            the dependent(y) variable specified

        Returns
        -------
        float
            calculated coefficient of determination(R²) of the polynomial regression
        """

        try:
            x = self.df_input[[independent]]
            y = self.df_input[[dependent]]

            poly = PolynomialFeatures(degree = 2)
            x_poly = poly.fit_transform(x)   

            model = LinearRegression()
            model.fit(x_poly, y)
            y_poly_pred = model.predict(x_poly)
            r2 = r2_score(y,y_poly_pred)
            return round(r2, 4)
        except Exception as e:
            print(e)

    def get_slope(self, independent, dependent, second_indep=None):
        """

        Returns the slope of the regression

        Returns the calculated slope(m) of the simple linear regression given that there is no second independent variable specified, else it will return a list containing the calculated slope(m) of the multiple linear regression
        
        Parameters
        ----------
        independent : str
            the independent(x) variable specified
        dependent : str
            the dependent(y) variable specified
        second_indep : str, optional
            the second independent(x) variable specified used for multiple linear regression
        
        Returns
        -------
        float
            calculated slope(m) of the simple linear regression
        list
            alternatively returns the list of calculated slope(m) of the multiple linear regression
        """

        try:
            if second_indep is None:
                x = self.df_input[[independent]]
                y = self.df_input[[dependent]]

                x = sm.add_constant(x)
                model = sm.OLS(y, x).fit() 
                return round(model.params[independent], 4)
            else:
                x = self.df_input[[independent, second_indep]]
                y = self.df_input[[dependent]]

                x = sm.add_constant(x)
                model = sm.OLS(y, x).fit() 
                coef_df = model.params
                return coef_df
        except Exception as e:
            print(e)

    def get_intercept(self, independent, dependent):
        """

        Returns the calculated intercept of the simple linear regression
        
        Parameters
        ----------
        independent : str
            the independent(x) variable specified
        dependent : str
            the dependent(y) variable specified

        Returns
        -------
        float
            intercept of the simple linear regression
        """

        try:
            x = self.df_input[[independent]]
            y = self.df_input[[dependent]]

            lm = LinearRegression()
            lm.fit(x, y)
            b = lm.intercept_
            return round(b[0], 4)
        except Exception as e:
            print(e)

    def get_coeff_det(self, independent, dependent, second_indep=None):
        """

        Returns the calculated coefficient of determination(R²) of the regression

        Returns the calculated coefficient of determination(R²) of the simple linear regression given that there is no second independent variable specified, else it will return the calculated coefficient of determination(R²) of the mulltiple linear regression 
        
        Parameters
        ----------
        independent : str
            the independent(x) variable specified
        dependent : str
            the dependent(y) variable specified
        second_indep : str, optional
            the second independent(x) variable specified used for multiple linear regression
        
        Returns
        -------
        float
            coefficient of determination(R²) of the regression
        """

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

    def get_adj_coeff_det(self, independent, dependent, second_indep=None):
        """
        
        Returns the calculated adjusted coefficient of determination(R²) of the regression

        returns the calculated adjusted coefficient of determination(R²) of the simple linear regression given that there is no second independent variable specified, else it will return the calculated adjusted coefficient of determination(R²) of the mulltiple linear regression 
        
        Parameters
        ----------
        independent : str
            the independent(x) variable specified
        dependent: str
            the dependent(y) variable specified
        second_indep: str, optional
            the second independent(x) variable specified used for multiple linear regression

        Returns
        -------
        float
            calculated adjusted coefficient of determination(R²) of the regression
        """

        try:
            if second_indep is None:
                if isinstance(independent, str) and isinstance(dependent, str):
                    x = self.df_input[[independent]]
                    y = self.df_input[[dependent]]
                elif isinstance(independent, pd.DataFrame) and isinstance(dependent, pd.DataFrame):
                    x = independent
                    y = dependent                  
            else:
                x = self.df_input[[independent, second_indep]]
                y = self.df_input[[dependent]]

            x = sm.add_constant(x)           
            model = sm.OLS(y, x).fit()
            return round(model.rsquared_adj, 4)
        except Exception as e:
            print(e)

    def get_pvalue(self, independent, dependent, second_indep=None):
        """

        Returns the calculated P-value/s of the regression

        Returns the dataframe containing calculated P-value/s of the simple linear regression given that there is no second independent variable specified, else it will return the calculated P-value/s of the mulltiple linear regression 
        
        Parameters
        ----------
        independent : str
            the independent(x) variable specified
        dependent : str
            the dependent(y) variable specified
        second_indep : str, optional
            the second independent(x) variable specified used for multiple linear regression
        
        Returns
        -------
        pandas Dataframe
            dataframe containing the P-value/s of the regression
        """

        try:
            if second_indep is None:
                if isinstance(independent, str) and isinstance(dependent, str):
                    x = self.df_input[[independent]]
                    y = self.df_input[[dependent]]
                elif isinstance(independent, pd.DataFrame) and isinstance(dependent, pd.DataFrame):
                    x = independent
                    y = dependent                  
            else:
                x = self.df_input[[independent, second_indep]]
                y = self.df_input[[dependent]]

            x = sm.add_constant(x)           
            model = sm.OLS(y, x).fit()
            pvalue = model.pvalues
            return pvalue
        except Exception as e:
            print(e)

    def line_eq(self, independent, dependent):
        """

        Returns the line equation of the simple linear regression 
        
        Parameters
        ----------
        independent : str
            the independent(x) variable specified
        dependent : str
            the dependent(y) variable specified
        
        Returns
        -------
        str
            line equation of the simple linear regression
        """

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

    def poly_eq(self, independent, dependent):
        """

        Returns the equation of the polynomial regression

        Parameters
        ----------
        independent : str
            the independent(x) variable specified
        dependent : str
            the dependent(y) variable specified
        
        Returns
        -------
        str
            line equation of the polynomial regression
        """

        try:
            x = self.df_input[[independent]]
            y = self.df_input[[dependent]]

            poly = PolynomialFeatures(degree = 2)
            x_poly = poly.fit_transform(x)  

            model = LinearRegression()
            model.fit(x_poly, y)
            coef_arr = model.coef_
            intercept_arr = model.intercept_
            
            poly_equation = "y = " + str(round(coef_arr[0][2], 4)) + "x\xB2"
            
            if(coef_arr[0][1] < 0):
                poly_equation += " + (" + str(round(coef_arr[0][1], 4)) + "x" + ")"
            else:
                poly_equation += " + " + str(round(coef_arr[0][1], 4)) + "x"
            
            if(intercept_arr[0] < 0):
                poly_equation += " + (" + str(round(intercept_arr[0], 4)) + ")"
            else:
                poly_equation += " + " + str(round(intercept_arr[0], 4))
           
            return  poly_equation
        except Exception as e:
            print(e)

    def scatter_plot(self, independent, dependent, second_indep=None):
        """

        Generates the visualization of the scatter plot

        Generates the 2D visualization of scatter plot given that no second independent variable is specified, else it will generate a 3D visualization

        Parameters
        ----------
        independent : str
            the independent(x) variable specified
        dependent : str
            the dependent(y) variable specified
        second_indep : str, optional
            the second independent(x) variable specified used for the plot
        
        Returns
        -------
        figure
            visualization of the scatter plot
        """

        try:
            if second_indep is None:
                x = self.df_input[independent]
                y = self.df_input[dependent]

                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.scatter(x, y, color = 'red')
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
                ax.scatter(x, y, z, color = 'red')
                ax.set_xlabel(independent)
                ax.set_ylabel("Number of cases of " + dependent)
                ax.set_zlabel(second_indep)
                ax.axis('tight')
                plt.title("Scatter Plot of " + dependent + ", " + independent + " and " + second_indep)
                plt.show()
        except Exception as e:
            print(e)

    def linear_regression(self, independent, dependent):
        """

        Generates the visualization for simple linear regression

        Parameters
        ----------
        independent : str
            the independent(x) variable specified
        dependent : str
            the dependent(y) variable specified
        
        Returns
        -------
        figure
            visualization of the simple linear regression
        """

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
            ax.legend(fontsize=9, loc="upper right")
            ax.scatter(x, y, color = 'red')
            ax.set_xlabel(independent)
            ax.set_ylabel(dependent)
            ax.axis('tight')
            plt.title("Linear Regression of " + independent + " and " + dependent)
            plt.show()
                
        except Exception as e:
            print(e)

    def linear_reg_summary(self, independent, dependent, second_indep=None):
        """

        Generates the calculated statistical values of the regression

        Generates the calculated statistical values for the linear regression such as the standard error, coefficient of determination(R²) and p-value, in table form

        Parameters
        ----------
        independent : str
            the independent(x) variable specified
        dependent : str
            the dependent(y) variable specified
        second_indep : str, optional
            the second independent(x) variable specified used for the multiple linear regression
        
        Returns
        -------
        statsmodels.summary
            table summary containing the calculated statistical values of the regression
        """

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

    def polynomial_reg(self, independent, dependent):
        """

        Generates the visualization for polynomial regression

        Parameters
        ----------
        independent : str
            the independent(x) variable specified
        dependent : str
            the dependent(y) variable specified

        Returns
        -------
        figure
            visualization of the polynomial regression
        """

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
            plt.legend(fontsize=9, loc="upper right")

            plt.title("Polynomial Regression of " + independent + " and " + dependent)
            plt.show()
                
        except Exception as e:
            print(e)

    def polynomial_reg_summary(self, independent, dependent):
        """

        Generates the calculated value of the coefficient of determination(R²) of the polynomial regression 

        Parameters
        ----------
        independent : str
            the independent(x) variable specified
        dependent : str
            the dependent(y) variable specified
        second_indep : str, optional
            the second independent(x) variable specified used for the multiple linear regression

        Parameters
        ----------
        str
            calculated value of the coefficient of determination(R²) of the polynomial regression 
        """

        try:
            poly_coeff_det = self.get_poly_coeff_det(independent, dependent)

            print("R\xb2 of the polynomial regression of " + independent + " and " + dependent + ": " + str(poly_coeff_det))

        except Exception as e:
            print(e)

    def linear_reg_table(self, title, independent, dependent, second_indep=None):
        """

        Generates the summary of the calculated values for P-value, correlation coefficient, coefficient of determination(R²) and adjusted coefficient of determination of regression, in table form

        Parameters
        ----------
        title : str
            title of the column for dependent(y) variables
        independent : str
            the independent(x) variable specified
        dependent : list
            list containing the string value of the dependent(y) variables
        second_indep : str, optional
            the second independent(x) variable specified used for the multiple linear regression

        Returns
        -------
        pandas Dataframe
            summary of the calculated values for P-value, correlation coefficient, coefficient of determination(R²) and adjusted coefficient of determination of regression
        """

        try:
            if second_indep is None:
                coeff = []
                coeff_det = []
                adj_coeff_det = []
                pvalue = []
                total_occ = []

                for step in dependent:
                    coeff.append(self.get_slope(independent, step))
                    coeff_det.append(self.get_coeff_det(independent, step))
                    adj_coeff_det.append(self.get_adj_coeff_det(independent, step))
                    pvalue_df = self.get_pvalue(independent, step)
                    pvalue.append(round(pvalue_df.loc[independent], 4))
                    total_occ.append(self.df_input[step].sum())

                table_content =  {title: dependent, "Number of Cases": total_occ, "P-Value of " + independent: pvalue, "Coefficient of " + independent: coeff, "Coefficient of Determination (R^2)": coeff_det, "Adjusted Coefficient of Determination (R^2)": adj_coeff_det}
                table_df = pd.DataFrame(table_content)
                return table_df
            else:
                first_coeff = []
                second_coeff = []
                coeff_det = []
                adj_coeff_det = []
                first_pvalue = []
                second_pvalue = []
                total_occ = []

                for step in dependent:
                    if("†" in step):
                        step = step.replace("†,", "† ,")
                    elif("â€" in step):
                        step = step.replace("â€,", "â€ ,")
                    m = self.get_slope(independent, step, second_indep)
                    first_coeff.append(round(m[independent], 4))
                    second_coeff.append(round(m[second_indep], 4))
                    coeff_det.append(self.get_coeff_det(independent, step, second_indep))
                    adj_coeff_det.append(self.get_adj_coeff_det(independent, step, second_indep))
                    pvalue_df = self.get_pvalue(independent, step, second_indep)
                    first_pvalue.append(round(pvalue_df.loc[independent], 4))
                    second_pvalue.append(round(pvalue_df.loc[second_indep], 4))
                    total_occ.append(self.df_input[step].sum())

                table_content =  {title: dependent, "Number of Cases": total_occ, "P-Value of " + independent: first_pvalue, "P-Value of " + second_indep: second_pvalue, "Coefficient of " + independent: first_coeff, "Coefficient of " + second_indep: second_coeff, "Coefficient of Determination (R^2)": coeff_det, "Adjusted Coefficient of Determination (R^2)": adj_coeff_det}
                table_df = pd.DataFrame(table_content)
                return table_df
        except Exception as e:
            print(e)


    def nb_feature_select(self,estimator, X, y,cv_kfold=5):

        """
        Select the best features and cross-validated selection of best number of features using recursive feature elimination

        Parameters
        ----------
        estimator : object
            A supervised learning estimator that can provide feature importance either through a ``coef_``
            attribute or through a ``feature_importances_`` attribute.

        X : pandas DataFrame
            The features to be selected

        y : pandas DataFrame
            The target feature as a basis of feature importance
        cv_kfold : int (default=5)
            The number of folds/splits for cross validation

        Returns
        -------
        numpy array
            The selected features
        """

        try:
            selector = RFECV(estimator, step=1,cv=cv_kfold)
            selector = selector.fit(X,y)
            support = selector.support_
            selected = []
            for a, s in zip(X.columns, support):
                if(s):
                    selected.append(a)
            return selected
        except Exception as e:
            print(e)

    def naive_bayes(self,X_columns, y_column, cv_kfold=10, class_bins=0, bin_strat='uniform', feature_selection=True):

        """
        Perform naive Bayes (Gaussian) classification

        Parameters
        ----------
        X_columns : numpy array
            Column names of the predictor features 
        y_column : str
            Column name of the target feature
        cv_kfold : int
            The number of folds/splits for cross validation
        class_bins : int (default=0)
            The number of bins to class the target feature 
        bin_strat : {'uniform', 'quantile', 'kmeans'}, (default='uniform')
            Strategy of defining the widths of the bins

            uniform:
                All bins have identical widths
            quantile:
                All bins have the same number of points
            kmeans:
                Values of each bins have the same k-means cluster centroid
            feature_selection : binary (default=True)
                Determines if nb_feature_select is to be applied
        
        Returns
        -------
        numpy array
            True values of the target feature
        numpy array
            Predicted values of the target feature
        float
            Accuracy of the model (based on `balanced_accuracy_score<https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html>`_).
        """
        
        try:

            valid_strategy = ('uniform', 'quantile', 'kmeans')
            if bin_strat not in valid_strategy:
                raise ValueError("Valid options for 'bin_strat' are {}. "
                             "Got strategy={!r} instead."
                             .format(valid_strategy, bin_strat))
            valid_feature_selection = {1,0}

            if feature_selection not in valid_feature_selection:
                raise ValueError("Valid options for 'bin_strat' are {}. "
                             "Got strategy={!r} instead."
                             .format(valid_feature_selection, feature_selection))

            X = self.df_input[X_columns]
            y = self.df_input[[y_column]]

            scaler = MinMaxScaler()
            for col in X.columns:
                X[col] = scaler.fit_transform(X[[col]].astype(float))

            if(class_bins!=0):
                est = KBinsDiscretizer(n_bins=class_bins, encode='ordinal', strategy='kmeans')
                if(bin_strat=='percentile'):
                    est = KBinsDiscretizer(n_bins=class_bins, encode='ordinal', strategy='percentile')
                elif(bin_strat=='uniform'):
                    est = KBinsDiscretizer(n_bins=class_bins, encode='ordinal', strategy='uniform')
                y[[y.columns[0]]] = est.fit_transform(y[[y.columns[0]]])

            if(feature_selection):
                X = X[self.nb_feature_select(LogisticRegression(solver='lbfgs', multi_class='auto'), X, y, cv_kfold=10)]
                
            X = X.values.tolist()

            y = y[y.columns[0]].values.tolist()

            kf = KFold(n_splits=cv_kfold)

            y_true_values,y_pred_values = [], []

            nb = GaussianNB()
            accuracy = 0

            for train_index, test_index in kf.split(X,y):

                X_test = [X[ii] for ii in test_index]
                X_train = [X[ii] for ii in train_index]
                y_test = [y[ii] for ii in test_index]
                y_train = [y[ii] for ii in train_index]

                y_true_values = np.append(y_true_values, y_test)

                nb.fit(X_train,y_train)
                y_pred =nb.predict(X_test)
                y_pred_values = np.append(y_pred_values, y_pred)

                accuracy = np.around(balanced_accuracy_score(y_true_values, y_pred_values),decimals=4)

            return y_true_values, y_pred_values, accuracy
            


        except Exception as e:
            print(e)

    def naive_bayes_cm(self,X_columns, y_column,cv_kfold=10, class_bins=0, bin_strat='uniform', feature_selection=True):

        """
        Perform naive Bayes (Gaussian) classification and visualize the results

        Parameters
        ----------
        X_columns : numpy array
            Column names of the predictor features 
        y_column : str
            Column name of the target feature
        cv_kfold : int
            The number of folds/splits for cross validation
        class_bins : int (default=0)
            The number of bins to class the target feature 
        bin_strat : {'uniform', 'quantile', 'kmeans'}, (default='uniform')
            Strategy of defining the widths of the bins

            uniform:
                All bins have identical widths
            quantile:
                All bins have the same number of points
            kmeans:
                Values of each bins have the same k-means cluster centroid
            feature_selection : binary (default=True)
                Determines if nb_feature_select is to be applied
        
        Returns
        -------
        figure
            Visualization (confusion matrix) of the naive Bayes classifier 

        """

        try:

            valid_strategy = ('uniform', 'quantile', 'kmeans')
            if bin_strat not in valid_strategy:
                raise ValueError("Valid options for 'bin_strat' are {}. "
                             "Got bin_strat={!r} instead."
                             .format(valid_strategy, bin_strat))
            valid_feature_selection = {True,False}
            
            if feature_selection not in valid_feature_selection:
                raise ValueError("Valid options for 'bin_strat' are {}. "
                             "Got feature_selection={!r} instead."
                             .format(valid_feature_selection, feature_selection))            

            y_true, y_pred, accuracy = self.naive_bayes(X_columns, y_column, cv_kfold=cv_kfold, class_bins=class_bins, bin_strat=bin_strat, feature_selection=feature_selection)
            cm = confusion_matrix(y_true, y_pred)
            cm_norm = (cm.astype('float') / cm.sum(axis=1)[:, np.newaxis])*100
            
            ticks = []

            if(class_bins!=0):
                y = self.df_input[[y_column]]
                est = KBinsDiscretizer(n_bins=class_bins, encode='ordinal', strategy='kmeans')
                if(bin_strat=='percentile'):
                    est = KBinsDiscretizer(n_bins=class_bins, encode='ordinal', strategy='percentile')
                elif(bin_strat=='uniform'):
                    est = KBinsDiscretizer(n_bins=class_bins, encode='ordinal', strategy='uniform')
                new_y = est.fit_transform(y[[y.columns[0]]])
                new_df = pd.DataFrame(new_y)
                edges = est.bin_edges_[0]
                new_df = pd.concat([new_df,y],axis=1)
                first = True
                for bins in new_df[0].unique():
                    if (first):
                        ticks.append(str(int(round(edges[int(bins)])))+" - "+str(int(round(edges[int(bins+1)]))))
                        first = False
                    else:
                        ticks.append(str(int(round(edges[int(bins)]))+1)+" - "+str(int(round(edges[int(bins+1)]))))


            fig, ax = plt.subplots()
            im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            ax.figure.colorbar(im, ax=ax)

            ax.set(xticks=np.arange(cm.shape[1]),yticks=np.arange(cm.shape[0]),ylabel='True Label',xlabel='Predicted Label')

            thresh = cm.max() / 2
            for x in range(cm_norm.shape[0]):
                for y in range(cm_norm.shape[1]):
                    if(x==y):
                        ax.text(y,x,f"{cm[x,y]}({cm_norm[x,y]:.2f}%)",ha="center", va="center", fontsize=12, color="white" if cm[x, y] > thresh else "black")
                    else:
                        ax.text(y,x,f"{cm[x,y]}({cm_norm[x,y]:.2f}%)",ha="center", va="center", color="white" if cm[x, y] > thresh else "black")
            ax.annotate("Accuracy: "+ str(accuracy),xy=(0.25, 0.9), xycoords='figure fraction')
            if(class_bins!=0):
                plt.xticks(np.arange(cm.shape[1]),ticks)
                plt.yticks(np.arange(cm.shape[0]),ticks)
            plt.title("Naive Bayes Confusion Matrix ("+y_column+")", y=1.05)
            plt.subplots_adjust(left=0)
            plt.show()
        except Exception as e:
                print(e)

    def km_feature_select(self,X_columns, y_column, k, n_features = 2):         

        """
        Select the best n number of features using silhouette analysis and forward stepwise regression

        Parameters
        ----------
        X_columns : numpy array
            Column names of the predictor features 
        y_column : str
            Column name of the target feature
        k : int
            number of clusters 
        n_features : int (default=2)
            number of features to be selected
        Returns
        -------
        numpy array
            The selected features

        """

        try:
            X = self.df_input[X_columns]
            y = self.df_input[[y_column]]
            features = pd.concat([X,y],axis=1)
            features_selected = [y_column]
            while(n_features>1):
                temp_selected = ""
                temp_coef = 0
                for col in X_columns:
                    temp_feat_sel = np.append(features_selected,col)
                    kmeans_model = KMeans(n_clusters=k,random_state=0).fit(features[temp_feat_sel])
                    labels = kmeans_model.labels_
                    sil_coef = silhouette_score(features[temp_feat_sel], labels, metric='euclidean')
                    if((col not in features_selected) and (sil_coef>temp_coef)):
                        temp_coef = sil_coef
                        temp_selected = col
                features_selected = np.append(features_selected,temp_selected)
                n_features -= 1
            return features_selected

        except Exception as e:
                    print(e)

    def k_means(self,X_columns, y_column, k, n_features=2):
        
        """
        Perform k-means clustering

        Parameters
        ----------
        X_columns : numpy array
            Column names of the predictor features 
        y_column : str
            Column name of the target feature
        k : int
            number of clusters 
        n_features : int (default=2)
            number of features to be selected
        
        Returns
        -------
        centroids,sil_coef,labeled_features
        numpy array
            Centroids of the clusters generated
        float
            Predicted values of the target feature
        pandas Dataframe
            The dataset with the cluster labels 
        """     

        try:
            X = self.df_input[X_columns]
            y = self.df_input[[y_column]]
            features = pd.concat([X,y],axis=1)
            scaler = MinMaxScaler()
            for col in features.columns:
                features[col] = scaler.fit_transform(features[[col]].astype(float))
            if(n_features!=0):
                sel_feat = self.km_feature_select(X_columns, y_column, 5, n_features =n_features)
                features = features[sel_feat]
            kmeans_model = KMeans(n_clusters=k,random_state=0).fit(features)
            labels = kmeans_model.labels_
            sil_coef = np.around(silhouette_score(features, labels, metric='euclidean'), decimals=4)
            centroids = np.around(kmeans_model.cluster_centers_, decimals=4)
            labels_df = pd.DataFrame(data=labels)
            labels_df.columns = ['clusters']
            labeled_features= pd.concat([features,labels_df], axis=1)

            return centroids,sil_coef,labeled_features

        except Exception as e:
            print(e)

    def k_means_cc(self, X_columns, y_column, k, n_features=0):
        """
        Perform k-means clustering and visualize the results

        Parameters
        ----------
        X_columns : numpy array
            Column names of the predictor features 
        y_column : str
            Column name of the target feature
        k : int
            number of clusters 
        n_features : int (default=2)
            number of features to be selected
        
        Returns
        -------
        figure
            Scatter plots and the silhouette coefficient
        figure
            Centroid Chart of the cluster centroids
        """
        try:
            centroids, sil_coef,labeled_features = self.k_means(X_columns, y_column, k, n_features=n_features)
            

            fig = plt.figure(1)
            fig.subplots_adjust(hspace=0.5)
            for x in range(0,centroids.shape[1]):
                ax = fig.add_subplot(round(centroids.shape[1]/2), 2, x+1)
                cmap = 'Set1'
                for c in range(centroids.shape[0]):
                    temp_df = labeled_features[labeled_features['clusters'] == c]
                    ax.scatter(temp_df[labeled_features.columns[x]], temp_df[labeled_features.columns[0]], label=c, cmap=cmap[c])
                ax.set_xlabel(labeled_features.columns[x])
                ax.axis('tight')
                if(x == 0):
                    plt.legend(title='Clusters', loc='upper right',bbox_to_anchor=(-0.1, 1.1))
            fig.text(0.04, 0.5, labeled_features.columns[0], va='center', rotation='vertical')
            plt.annotate("Silhouette Coefficient: "+ str(sil_coef),xy=(10, 10), xycoords='figure pixels')
            plt.suptitle("Scatter Plot of "+labeled_features.columns[0]+" against other Features")
                       
            fig2 = plt.figure(2)
            ax2 = fig2.subplots()
            for k in range(centroids.shape[0]):
                plt.plot(range(centroids.shape[1]),centroids[k], label=str(k)+": "+str(centroids[k]))
            plt.xticks(range(centroids.shape[1]),labeled_features.columns[:-1])
            plt.xlabel("Features")
            plt.ylabel("Location")
            plt.setp( ax2.xaxis.get_majorticklabels(), rotation=-15, ha="left" )
            plt.legend(loc="upper right",title='Clusters (Centroids)')
            plt.title("Centroid Chart ("+y_column+")")
            plt.tight_layout()

            plt.show()

        except Exception as e:
=======
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
>>>>>>> 77e746da2551d1dd1065fadbfff4c05467b28921
                print(e)