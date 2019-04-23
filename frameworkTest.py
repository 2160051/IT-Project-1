import os
import warnings
import pandas as pd
from framework import DataAnalyticalFramework
# warnings.simplefilter(action='ignore', category=FutureWarning)
# warnings.simplefilter(action='ignore', category=RuntimeWarning)

file_path = os.path.dirname(__file__) + "/Data Analytics.csv"
fig_inp = ''

while fig_inp != 15 and fig_inp != 'N':
    print("----------Select Type of Figure-----------")
    print("**Scatter Plot**")
    print("1 Simple Scatter Plot")
    print("2 Scatter Plot with Multiple Attributes")
    print()
    print("**Linear Regression**")
    print("3 Simple Linear Regression")
    print("4 Linear Regression, Results Only")
    print("5 Multiple Linear Regression")
    print("6 Table Summary Simple Linear Regression")
    print("7 Table Summary Multiple Linear Regression")
    print()
    print("**Polynomial Curve**")
    print("8 Polynomial Curve")
    print("9 Polynomial Curve, Results Only")
    print()
    print("**Naive-Bayes**")
    print("10 Applying Naive-Bayes #defunct")
    print("11 Applying Naive-Bayes, Hypothesis Result Only #defunct")
    print()
    print("**K-means Clustering**")
    print("12 K-means #defunct")
    print("13 K-means, Result Only #defunct")
    print()
    print("14 Help")
    print("15 Quit")
    print("----------Input-----------")
    print("Enter the number (#) of the chosen type of figure: ")
    fig_inp = int(input())
    os.system("cls")

    if(fig_inp == 14):
        print("----------Help-----------")
        print("1 Simple Scatter Plot")
        print("2 Scatter Plot with Multiple Attributes")
        print("3 Simple Linear Regression")
        print("4 Multiple Linear Regression")
        print("5 Polynomial Curve")
        print("6 Linear Regression, Results Only")
        print("7 Applying Naive-Bayes #defunct")
        print("8 Applying Naive-Bayes, Hypothesis Result Only #defunct")
        print("9 K-means #defunct")
        print("10 K-means, Result Only #defunct")
        print("11 Back to Start")
        print("12 Quit")
        print("----------Input-----------")
        print("Enter the number (#) of the chosen type of figure: ")
        help_inp = int(input())
        os.system("cls")

        while(help_inp < 0 or help_inp > 12):
            print("User sent a wrong input. Please try again.")
            print("----------Help-----------")
            print("1 Simple Scatter Plot")
            print("2 Scatter Plot with Multiple Attributes")
            print("3 Simple Linear Regression")
            print("4 Multiple Linear Regression")
            print("5 Polynomial Curve")
            print("6 Linear Regression, Results Only")
            print("7 Applying Naive-Bayes #defunct")
            print("8 Applying Naive-Bayes, Hypothesis Result Only #defunct")
            print("9 K-means #defunct")
            print("10 K-means, Result Only #defunct")
            print("11 Back to Start")
            print("12 Quit")
            print("----------Input-----------")
            print("Enter the number (#) of the chosen type of figure: ")
            help_inp = int(input())
            os.system("cls")

        if(help_inp == 1):
            print("Scatter Plot - 2D data visualization which makes use of dots or points to represents the values of two different variables. It is sometimes referred to as correlation plot since it is used for showing the correlation or relationship between two variables.")
            print("Interpretation:")
            print("1. Positive Correlation - occurs when the values are increasing together. This is seen in a plot wherein the points are aligned to resemble a line leaning to the right.")
            print("2. Negative Correlation - occurs when the values are decreasing together. This is seen in a plot wherein the points are aligned to resemble a line leaning to the left.")
            print("3. No Correlation - occurs when the values are far from each other and has a shape that does not closely resembles a line.")
        elif(help_inp == 2):
            print("Scatter Plot with Multiple Attributes - 3D data visualization which makes use of dots or points to represents the values of three different variables. Instead of only making use of the x and y-axis, the z-axis is also used.")
            print("Interpretation:")
            print("1. Positive Correlation - occurs when the values are increasing together. This is seen in a plot wherein the points are aligned to resemble a line leaning to the right.")
            print("2. Negative Correlation - occurs when the values are decreasing together. This is seen in a plot wherein the points are aligned to resemble a line leaning to the left.")
            print("3. No Correlation - occurs when the values are far from each other and has a shape that does not closely resembles a line.")
        elif(help_inp == 3):
            print("Linear Regression - type of predictive analysis model for determining the relationship between two variables by fitting a line. One variable is the independent variable (x-axis) while the other is the dependent variable (y-axis). The analysis of a linear regression is based on its p-values and coefficients. This section covers only the visualization and the equation of the line of one independent variable and one dependent variable using linear regression. For calculating the result and hypothesis testing, please make use of the 5th choice in the type of figure which is 'Linear Regression, Results Only'.")
        elif(help_inp == 4):
            print("Multiple Linear Regression - type of predictive analysis model for determining the relationship between multiple variables by fitting a line. Unlike a simple linear regression, a multiple linear regression makes use of one dependent variable and two or more independent variable. For this program, only one dependent variable and two independent variables will be compared. The analysis is still based on the measured p-values and coefficients of the variables used.")
            print("Analysis:")
            print("1. P-value - of each independent variable tests the null hypothesis or the idea that the variable has no correlation with the dependent variable. It is used to determine the statistical significance of the correlation between the two variables. A low value, which is less than the indicated significance level or alpha (α), means that the null hypothesis can be rejected. On the other hand, a high p-value, more than the indicated significance level or alpha (α), means that the null hypothesis  can be accepted and that the changes in the independent variable does not affect the values of the dependent variable.")
            print("2. Coefficient - otherwise known as slope coefficient, it represents the mean change of the dependent variable as the independent variable changes. It is used for identifying the type of correlation between the variables, either positive or negative correlation. A positive correlation means that as the value of the independent variable increases, so does the mean of the dependent variable. The case is opposite in a negative correlation.")
        elif(help_inp == 5):
            print("Polynomial Curve - another way of analyzing regression wherein the relationship between the independent variable and dependent variable is modelled as a polynomial visualization of a certain specified degree. Unlike linear regression wherein the data visualization shows a line, polynomial regression takes the form of curves or other polynomial shapes.")
            print("Analysis:")
            print("1. P-value - of each independent variable tests the null hypothesis or the idea that the variable has no correlation with the dependent variable. It is used to determine the statistical significance of the correlation between the two variables. A low value, which is less than the indicated significance level or alpha (α), means that the null hypothesis can be rejected. On the other hand, a high p-value, more than the indicated significance level or alpha (α), means that the null hypothesis  can be accepted and that the changes in the independent variable does not affect the values of the dependent variable.")
            print("2. Coefficient - otherwise known as slope coefficient, it represents the mean change of the dependent variable as the independent variable changes. It is used for identifying the type of correlation between the variables, either positive or negative correlation. A positive correlation means that as the value of the independent variable increases, so does the mean of the dependent variable. The case is opposite in a negative correlation.")
        elif(help_inp == 6):
            print("Linear Regression - type of predictive analysis model for determining the relationship between two variables by fitting a line. One variable is the independent variable (x-axis) while the other is the dependent variable (y-axis). The analysis of a linear regression is based on its p-values and coefficients. This section covers only the visualization and the equation of the line of one independent variable and one dependent variable using linear regression. For displaying the visualization of the variables using simple linear regression, please make use of the 3rd choice in the type of figure which is 'Simple Linear Regression'.")
            print("Analysis:")
            print("1. P-value - of each independent variable tests the null hypothesis or the idea that the variable has no correlation with the dependent variable. It is used to determine the statistical significance of the correlation between the two variables. A low value, which is less than the indicated significance level or alpha (α), means that the null hypothesis can be rejected. On the other hand, a high p-value, more than the indicated significance level or alpha (α), means that the null hypothesis  can be accepted and that the changes in the independent variable does not affect the values of the dependent variable.")
            print("2. Coefficient - otherwise known as slope coefficient, it represents the mean change of the dependent variable as the independent variable changes. It is used for identifying the type of correlation between the variables, either positive or negative correlation. A positive correlation means that as the value of the independent variable increases, so does the mean of the dependent variable. The case is opposite in a negative correlation.")
        # elif(help_inp == 7):
        #     #code here
        # elif(help_inp == 8):
        #     #code here
        # elif(help_inp == 9):
        #     #code here
        elif(help_inp == 11):
            os.system("cls")
            continue
        elif(help_inp == 12):
            print("User has closed the program.")
            exit() 

        print("----------Input-----------")
        print("Would you like to continue <Y/N>? ")
        cont_inp = input()

        if(cont_inp == "Y"):
            os.system("cls")
            continue
        else:
            print("User has either closed the program or a wrong input was sent.")
            exit() 
    elif(fig_inp == 15):
        print("User has closed the program.")
        exit()
    elif(fig_inp < 0 or fig_inp > 15):
        print("User sent a wrong input. Please try again.")
        continue
    
    if(fig_inp != 6 and fig_inp != 7 and fig_inp !=10 and fig_inp !=11 and fig_inp !=12 and fig_inp !=13):
        print("----------Would You Like To...-----------")
        print("1 Consider All States and Choose Attribute to Compare")
        print("2 Choose a State and Attribute to Compare")
        print("3 Choose by Week and Compare")
        print("4 Choose by Week and by Category of Disease and Compare")
        print("5 Choose Consider All States, by Category of Disease and Compare")
        print("6 Choose a State and by Category of Disease and Compare")
        print("7 Back to Start")
        print("----------Input-----------")
        print("Enter the number (#) of your choice: ")
        choice_inp = int(input())
        os.system("cls")

        if(choice_inp == 7):
            os.system("cls")
            continue
    elif(fig_inp == 10 or fig_inp == 11 or fig_inp == 12 or fig_inp == 13):
        choice_inp = 0
    
        test_frame = DataAnalyticalFramework(file_path)
        disease_arr = test_frame.get_column(3, 54)
        attribute_arr = test_frame.get_column(54, 80)
        print("----------Select Notifiable Disease to Compare-----------")
        test_frame.print_arr(disease_arr)
        print(str(len(disease_arr) + 1) + " Back to Start")
        print("----------Input-----------")
        print("Enter the number (#) of your choice: ")
        disease_inp = int(input()) - 1
        
        if(disease_inp == len(disease_arr)):
            os.system("cls")
            continue
    



    if(choice_inp == 1):
        test_frame = DataAnalyticalFramework(file_path)
        test_frame.set_new_df(test_frame.group_frame_except("State", 3, 49))
        disease_arr = test_frame.get_column(1, 47)
        attribute_arr = test_frame.get_column(52, 78)

        print("----------Select Notifiable Disease to Compare-----------")
        test_frame.print_arr(disease_arr)
        print(str(len(disease_arr) + 1) + " Back to Start")
        print("----------Input-----------")
        print("Enter the number (#) of your choice: ")
        disease_inp = int(input()) - 1
        os.system("cls")

        if(disease_inp == len(disease_arr)):
            os.system("cls")
            continue

        #del attribute_arr[2 : 8]
        print("----------Select Attribute to Compare-----------")
        test_frame.print_arr(attribute_arr)
        print(str(len(attribute_arr) + 1) + " Back to Start")
        print("----------Input-----------")
        print("Enter the number (#) of your choice: ")
        attribute_inp = int(input()) - 1
        os.system("cls")

        if(attribute_inp == len(attribute_arr)):
            os.system("cls")
            continue
    elif(choice_inp == 2):
        test_frame = DataAnalyticalFramework(file_path)
        state_arr = test_frame.get_row("State", 0, 50)
        disease_arr = test_frame.get_column(3, 49)
        attribute_arr = test_frame.get_column(54, 80)   
    elif(choice_inp == 3):
        test_frame = DataAnalyticalFramework(file_path)
        disease_arr = test_frame.get_column(3, 49)
        attribute_arr = test_frame.get_column(54, 80)

        print("----------Select Week-----------")
        print("Select a week from week 1 - 26")
        print("27 Back to Start")
        print("----------Input-----------")
        print("Enter the week (#) of your choice: ")
        week_inp = int(input())
        os.system("cls")

        if(week_inp == 27):
            os.system("cls")
            continue
        
        test_frame.set_new_df(test_frame.locate("Week", week_inp))
        
        print("----------Select Notifiable Disease to Compare-----------")
        test_frame.print_arr(disease_arr)
        print(str(len(disease_arr) + 1) + " Back to Start")
        print("----------Input-----------")
        print("Enter the number (#) of your choice: ")
        disease_inp = int(input()) - 1
        #os.system("cls")

        if(disease_inp == len(disease_arr)):
            os.system("cls")
            continue

        print("----------Select Attribute to Compare-----------")
        test_frame.print_arr(attribute_arr)
        print(str(len(attribute_arr) + 1) + " Back to Start")
        print("----------Input-----------")
        print("Enter the number (#) of your choice: ")
        attribute_inp = int(input()) - 1
        os.system("cls")

        if(attribute_inp == len(attribute_arr)):
            os.system("cls")
            continue
    elif(choice_inp == 4):
        test_frame = DataAnalyticalFramework(file_path)
        disease_arr = test_frame.get_column(49, 54)
        attribute_arr = test_frame.get_column(54, 80)

        print("----------Select Week-----------")
        print("Select a week from week 1 - 26")
        print("27 Back to Start")
        print("----------Input-----------")
        print("Enter the week (#) of your choice: ")
        week_inp = int(input())
        os.system("cls")

        if(week_inp == 27):
            os.system("cls")
            continue

        print("----------Select Category of Disease to Compare-----------")
        test_frame.print_arr(disease_arr)
        print(str(len(disease_arr) + 1) + " Back to Start")
        print("----------Input-----------")
        print("Enter the number (#) of your choice: ")
        disease_inp = int(input()) - 1
        os.system("cls")

        if(disease_inp == len(disease_arr)):
            os.system("cls")
            continue

        #del attribute_arr[2 : 8]
        print("----------Select Attribute to Compare-----------")
        test_frame.print_arr(attribute_arr)
        print(str(len(attribute_arr) + 1) + " Back to Start")
        print("----------Input-----------")
        print("Enter the number (#) of your choice: ")
        attribute_inp = int(input()) - 1
        os.system("cls")

        if(attribute_inp == len(attribute_arr)):
            os.system("cls")
            continue
        
        test_frame.set_new_df(test_frame.locate("Week", week_inp))   
    elif(choice_inp == 5):
        test_frame = DataAnalyticalFramework(file_path)
        test_frame.set_new_df(test_frame.group_frame_except("State", 3, 49))
        disease_arr = test_frame.get_column(47, 52)
        attribute_arr = test_frame.get_column(52, 78)

        print("----------Select Category of Disease to Compare-----------")
        test_frame.print_arr(disease_arr)
        print(str(len(disease_arr) + 1) + " Back to Start")
        print("----------Input-----------")
        print("Enter the number (#) of your choice: ")
        disease_inp = int(input()) - 1
        os.system("cls")

        if(disease_inp == len(disease_arr)):
            os.system("cls")
            continue

        print("----------Select Attribute to Compare-----------")
        test_frame.print_arr(attribute_arr)
        print(str(len(attribute_arr) + 1) + " Back to Start")
        print("----------Input-----------")
        print("Enter the number (#) of your choice: ")
        attribute_inp = int(input()) - 1
        os.system("cls")

        if(attribute_inp == len(attribute_arr)):
            os.system("cls")
            continue  
    elif(choice_inp == 6):
        test_frame = DataAnalyticalFramework(file_path)
        state_arr = test_frame.get_row("State", 0, 50)
        disease_arr = test_frame.get_column(49, 54)
        attribute_arr = test_frame.get_column(54, 80)
        
    if(choice_inp == 2 or choice_inp == 6):
        print("----------Select State-----------")
        test_frame.print_arr(state_arr)
        print(str(len(state_arr) + 1) + " Back to Start")
        print("----------Input-----------")
        print("Enter the number (#) of your choice: ")
        state_inp = int(input()) - 1
        #os.system("cls")

        if(state_inp == len(state_arr)):
            os.system("cls")
            continue

        test_frame.set_new_df(test_frame.extract_row("State", state_arr[state_inp])) 

    if(choice_inp == 2):  
        print("----------Select Notifiable Disease to Compare-----------")
        test_frame.print_arr(disease_arr)
        print(str(len(disease_arr) + 1) + " Back to Start")
        print("----------Input-----------")
        print("Enter the number (#) of your choice: ")
        disease_inp = int(input()) - 1
        os.system("cls")

        if(disease_inp == len(disease_arr)):
            os.system("cls")
            continue

        print("----------Select Attribute to Compare-----------")
        test_frame.print_arr(attribute_arr)
        print(str(len(attribute_arr) + 1) + " Back to Start")
        print("----------Input-----------")
        print("Enter the number (#) of your choice: ")
        attribute_inp = int(input()) - 1
        os.system("cls")

        if(attribute_inp == len(attribute_arr)):
            os.system("cls")
            continue
    elif(choice_inp == 6):  
        print("----------Select Category of Disease to Compare-----------")
        test_frame.print_arr(disease_arr)
        print(str(len(disease_arr) + 1) + " Back to Start")
        print("----------Input-----------")
        print("Enter the number (#) of your choice: ")
        disease_inp = int(input()) - 1
        os.system("cls")

        if(disease_inp == len(disease_arr)):
            os.system("cls")
            continue

        print("----------Select Attribute to Compare-----------")
        test_frame.print_arr(attribute_arr)
        print(str(len(attribute_arr) + 1) + " Back to Start")
        print("----------Input-----------")
        print("Enter the number (#) of your choice: ")
        attribute_inp = int(input()) - 1
        os.system("cls")

        if(attribute_inp == len(attribute_arr)):
            os.system("cls")
            continue
    
    if(fig_inp == 1):
        test_frame.scatter_plot(attribute_arr[attribute_inp], disease_arr[disease_inp])
    elif(fig_inp == 2):
        new_attrib_arr = attribute_arr.copy()
        new_attrib_arr.pop(attribute_inp)

        print("----------Select Another Attribute to Compare-----------")
        test_frame.print_arr(new_attrib_arr)
        print(str(len(new_attrib_arr) + 1) + " Back to Start")
        print("----------Input-----------")
        print("Enter the number (#) of your choice: ")
        attribute_second_inp = int(input()) - 1
        os.system("cls")

        if(attribute_second_inp == len(new_attrib_arr)):
            os.system("cls")
            continue

        test_frame.scatter_plot(attribute_arr[attribute_inp], disease_arr[disease_inp], new_attrib_arr[attribute_second_inp])
    elif(fig_inp == 3):
        test_frame.linear_regression(attribute_arr[attribute_inp], disease_arr[disease_inp])
    elif(fig_inp == 4):
        test_frame.linear_reg_summary(attribute_arr[attribute_inp], disease_arr[disease_inp])
    elif(fig_inp == 5):
        new_attrib_arr = attribute_arr.copy()
        new_attrib_arr.pop(attribute_inp)

        print("----------Select Another Attribute to Compare-----------")
        test_frame.print_arr(new_attrib_arr)
        print(str(len(new_attrib_arr) + 1) + " Back to Start")
        print("----------Input-----------")
        print("Enter the number (#) of your choice: ")
        attribute_second_inp = int(input()) - 1
        os.system("cls")

        if(disease_inp == len(disease_arr)):
            os.system("cls")
            continue

        test_frame.linear_reg_summary(attribute_arr[attribute_inp], disease_arr[disease_inp],  new_attrib_arr[attribute_second_inp])
    elif(fig_inp == 6):
        test_frame = DataAnalyticalFramework(file_path)
        state_arr = test_frame.get_row("State", 0, 50)
        disease_arr = test_frame.get_column(49, 54)
        attribute_arr = test_frame.get_column(54, 72)

        print("----------Select State-----------")
        test_frame.print_arr(state_arr)
        print(str(len(state_arr) + 1) + " Back to Start")
        print("----------Input-----------")
        print("Enter the number (#) of your choice: ")
        state_inp = int(input()) - 1
        #os.system("cls")

        if(state_inp == len(state_arr)):
            os.system("cls")
            continue

        test_frame.set_new_df(test_frame.extract_row("State", state_arr[state_inp]))
        print("----------Select Attribute to Compare-----------")
        test_frame.print_arr(attribute_arr)
        print(str(len(attribute_arr) + 1) + " Back to Start")
        print("----------Input-----------")
        print("Enter the number (#) of your choice: ")
        attribute_inp = int(input()) - 1
        os.system("cls")

        if(attribute_inp == len(attribute_arr)):
            os.system("cls")
            continue

        person_to_person = ["Carbapenemase-producing carbapenem-resistant Enterobacteriaceae â€, Klebsiella spp.", "Chlamydia trachomatis infection", "Gonorrhea", "Hepatitis (viral, acute, by type)â€, B", "Hepatitis (viral, acute, by type)â€, C, Confirmed", "Hepatitis (viral, acute, by type)â€, C, Probable", "Invasive Pneumococcal Disease, Age LT 5â€, Confirmed", "Invasive Pneumococcal Disease, Age LT 5â€, Probable", "Invasive Pneumococcal Disease, all agesâ€, Confirmed","Invasive Pneumococcal Disease, all agesâ€, Probable", "Meningococcal disease, all serogroups", "Shigellosis", "Syphilis, primary and secondary"]
        droplet_spread = ["Haemophilus influenzae, invasive disease (all ages, all serotypes)", "Invasive Pneumococcal Disease, Age LT 5â€, Confirmed", "Invasive Pneumococcal Disease, Age LT 5â€, Probable", "Invasive Pneumococcal Disease, all agesâ€, Confirmed", "Invasive Pneumococcal Disease, all agesâ€, Probable", "Meningococcal disease, all serogroups", "Mumps", "Pertussis", "Rubella", "Rubella, congenital syndrome"]
        airborne = ["Coccidioidomycosis", "Legionellosis", "Mumps", "Pertussis"]
        vector_borne = ["Babesiosis", "Dengue Virus Infections, Dengueâ€", "Dengue Virus Infections, Severe Dengue", "Ehrlichiosis and Anaplasmosis, Anaplasma phagocytophilum infection", "Ehrlichiosis and Anaplasmosis, Ehrlichia chaffeensis infection", "Ehrlichiosis and Anaplasmosis, Ehrlichia ewingii infection", "Ehrlichiosis and Anaplasmosis, Undetermined Ehrlichiosis/Anaplasmosis", "Malaria", "Rabies, animal", "Spotted Fever Rickettsiosis, Confirmed", "Spotted Fever Rickettsiosis, Probable", "Tetanus", "Varicella morbidity", "West Nile virus diseaseâ€, Neuroinvasive", "West Nile virus diseaseâ€, Nonneuroinvasive", "Zika virus disease, non-congenital"]
        vehicle_borne = ["Campylobacteriosis", "Carbapenemase-producing carbapenem-resistant Enterobacteriaceae â€, Enterobacter spp.", "Carbapenemase-producing carbapenem-resistant Enterobacteriaceae â€, Escherichia coli", "Cryptosporidiosis", "Giardiasis", "Hepatitis (viral, acute, by type)â€, A", "Hepatitis (viral, acute, by type)â€, B", "Hepatitis (viral, acute, by type)â€, C, Confirmed", "Hepatitis (viral, acute, by type)â€, C, Probable", "Salmonellosis (excluding Paratyphoid fever andTyphoid fever)â€", "Shiga toxin-producing Escherichia coli", "Shigellosis", "Syphilis, primary and secondary", "Vibriosis (Any species of the family Vibrionaceae, other than toxigenic Vibrio cholerae O1 or O139), Confirmed", "Vibriosis (Any species of the family Vibrionaceae, other than toxigenic Vibrio cholerae O1 or O139), Probable"]
        categorized_disease = [person_to_person, droplet_spread, airborne, vector_borne, vehicle_borne]
        
        print("----------Select Category of Disease to Compare-----------")
        test_frame.print_arr(disease_arr)
        print(str(len(disease_arr) + 1) + " All Categories")
        print(str(len(disease_arr) + 2) + " Back to Start")
        print("----------Input-----------")
        print("Enter the number (#) of your choice: ")
        disease_inp = int(input())
        os.system("cls")

        if(disease_inp == 1):
            print(test_frame.linear_reg_table("Person-to-Person Transmission", attribute_arr[attribute_inp], categorized_disease[0]))
            print("----------Would You Like to Save it in a CSV File?-----------")
            print("1 Yes")
            print("2 No. Go Back to Start")
            print("----------Input-----------")
            print("Enter the number (#) of your choice: ")
            save_inp = int(input())
            os.system("cls")

            if(save_inp == 1):
                table_df = test_frame.linear_reg_table("Person-to-Person Transmission", attribute_arr[attribute_inp], categorized_disease[0])
                table_df.to_csv(os.path.dirname(__file__) + "/Person-to-Person Transmission and " + attribute_arr[attribute_inp] + ".csv")
            elif(save_inp == 2):
                os.system("cls")
                continue
        elif(disease_inp == 2):
            print(test_frame.linear_reg_table("Droplet Spread", attribute_arr[attribute_inp], categorized_disease[1]))
            
            print("----------Would You Like to Save it in a CSV File?-----------")
            print("1 Yes")
            print("2 No. Go Back to Start")
            print("----------Input-----------")
            print("Enter the number (#) of your choice: ")
            save_inp = int(input())
            os.system("cls")

            if(save_inp == 1):
                table_df = test_frame.linear_reg_table("Droplet Spread", attribute_arr[attribute_inp], categorized_disease[1])
                table_df.to_csv(os.path.dirname(__file__) + "/Droplet Spread and " + attribute_arr[attribute_inp] + ".csv")
            elif(save_inp == 2):
                os.system("cls")
                continue
        elif(disease_inp == 3):
            print(test_frame.linear_reg_table("Air-borne Transmission", attribute_arr[attribute_inp], categorized_disease[2]))
            
            print("----------Would You Like to Save it in a CSV File?-----------")
            print("1 Yes")
            print("2 No. Go Back to Start")
            print("----------Input-----------")
            print("Enter the number (#) of your choice: ")
            save_inp = int(input())
            os.system("cls")

            if(save_inp == 1):
                table_df = test_frame.linear_reg_table("Air-borne Transmission", attribute_arr[attribute_inp], categorized_disease[2])
                table_df.to_csv(os.path.dirname(__file__) + "/Air-borne Transmission and " + attribute_arr[attribute_inp] + ".csv")
            elif(save_inp == 2):
                os.system("cls")
                continue
        elif(disease_inp == 4):
            print(test_frame.linear_reg_table("Vector-borne Transmission", attribute_arr[attribute_inp], categorized_disease[3]))
            
            print("----------Would You Like to Save it in a CSV File?-----------")
            print("1 Yes")
            print("2 No. Go Back to Start")
            print("----------Input-----------")
            print("Enter the number (#) of your choice: ")
            save_inp = int(input())
            os.system("cls")

            if(save_inp == 1):
                table_df = test_frame.linear_reg_table("Vector-borne Transmission", attribute_arr[attribute_inp], categorized_disease[3])
                table_df.to_csv(os.path.dirname(__file__) + "/Vector-borne Transmission and " + attribute_arr[attribute_inp] + ".csv")
            elif(save_inp == 2):
                os.system("cls")
                continue
        elif(disease_inp == 5):
            print(test_frame.linear_reg_table("Vehicle-borne Transmission", attribute_arr[attribute_inp], categorized_disease[4]))
            
            print("----------Would You Like to Save it in a CSV File?-----------")
            print("1 Yes")
            print("2 No. Go Back to Start")
            print("----------Input-----------")
            print("Enter the number (#) of your choice: ")
            save_inp = int(input())
            os.system("cls")

            if(save_inp == 1):
                table_df = test_frame.linear_reg_table("Vehicle-borne Transmission", attribute_arr[attribute_inp], categorized_disease[4])
                table_df.to_csv(os.path.dirname(__file__) + "/Vehicle-borne Transmission and " + attribute_arr[attribute_inp] + ".csv")
            elif(save_inp == 2):
                os.system("cls")
                continue
        elif(disease_inp == 6):
            print(test_frame.linear_reg_table("Person-to-Person Transmission", attribute_arr[attribute_inp], categorized_disease[0]))
            print()
            print(test_frame.linear_reg_table("Droplet Spread", attribute_arr[attribute_inp], categorized_disease[1]))
            print()
            print(test_frame.linear_reg_table("Air-borne Transmission", attribute_arr[attribute_inp], categorized_disease[2]))
            print()
            print(test_frame.linear_reg_table("Vector-borne Transmission", attribute_arr[attribute_inp], categorized_disease[3]))
            print()
            print(test_frame.linear_reg_table("Vehicle-borne Transmission", attribute_arr[attribute_inp], categorized_disease[4]))
            

            print("----------Would You Like to Save it in a CSV File?-----------")
            print("1 Yes")
            print("2 No. Go Back to Start")
            print("----------Input-----------")
            print("Enter the number (#) of your choice: ")
            save_inp = int(input())
            os.system("cls")

            if(save_inp == 1):
                table_df = test_frame.linear_reg_table("Person-to-Person Transmission", attribute_arr[attribute_inp], categorized_disease[0])
                table_df.to_csv(os.path.dirname(__file__) + "/Person-to-Person Transmission and " + attribute_arr[attribute_inp] + ".csv")

                table_df = test_frame.linear_reg_table("Droplet Spread", attribute_arr[attribute_inp], categorized_disease[1])
                table_df.to_csv(os.path.dirname(__file__) + "/Droplet Spread and " + attribute_arr[attribute_inp] + ".csv")

                table_df = test_frame.linear_reg_table("Air-borne Transmission", attribute_arr[attribute_inp], categorized_disease[2])
                table_df.to_csv(os.path.dirname(__file__) + "/Air-borne Transmission and " + attribute_arr[attribute_inp] + ".csv")

                table_df = test_frame.linear_reg_table("Vector-borne Transmission", attribute_arr[attribute_inp], categorized_disease[3])
                table_df.to_csv(os.path.dirname(__file__) + "/Vector-borne Transmission and " + attribute_arr[attribute_inp] + ".csv")

                table_df = test_frame.linear_reg_table("Vehicle-borne Transmission", attribute_arr[attribute_inp], categorized_disease[4])
                table_df.to_csv(os.path.dirname(__file__) + "/Vehicle-borne Transmission and " + attribute_arr[attribute_inp] + ".csv")
            elif(save_inp == 2):
                os.system("cls")
                continue
        elif(disease_inp == len(disease_arr) + 2):
            os.system("cls")
            continue
    elif(fig_inp == 7):
        test_frame = DataAnalyticalFramework(file_path)
        state_arr = test_frame.get_row("State", 0, 50)
        disease_arr = test_frame.get_column(49, 54)
        attribute_arr = test_frame.get_column(54, 72)

        print("----------Select State-----------")
        test_frame.print_arr(state_arr)
        print(str(len(state_arr) + 1) + " Back to Start")
        print("----------Input-----------")
        print("Enter the number (#) of your choice: ")
        state_inp = int(input()) - 1
        #os.system("cls")

        if(state_inp == len(state_arr)):
            os.system("cls")
            continue

        test_frame.set_new_df(test_frame.extract_row("State", state_arr[state_inp]))

        print("----------Select Attribute to Compare-----------")
        test_frame.print_arr(attribute_arr)
        print(str(len(attribute_arr) + 1) + " Back to Start")
        print("----------Input-----------")
        print("Enter the number (#) of your choice: ")
        attribute_inp = int(input()) - 1
        os.system("cls")

        if(attribute_inp == len(attribute_arr)):
            os.system("cls")
            continue

        new_attrib_arr = attribute_arr.copy()
        new_attrib_arr.pop(attribute_inp)

        print("----------Select Another Attribute to Compare-----------")
        test_frame.print_arr(new_attrib_arr)
        print(str(len(new_attrib_arr) + 1) + " Back to Start")
        print("----------Input-----------")
        print("Enter the number (#) of your choice: ")
        attribute_second_inp = int(input()) - 1
        os.system("cls")

        if(attribute_second_inp == len(new_attrib_arr)):
            os.system("cls")
            continue
        
        person_to_person = ["Carbapenemase-producing carbapenem-resistant Enterobacteriaceae â€, Klebsiella spp.", "Chlamydia trachomatis infection", "Gonorrhea", "Hepatitis (viral, acute, by type)â€, B", "Hepatitis (viral, acute, by type)â€, C, Confirmed", "Hepatitis (viral, acute, by type)â€, C, Probable", "Invasive Pneumococcal Disease, Age LT 5â€, Confirmed", "Invasive Pneumococcal Disease, Age LT 5â€, Probable", "Invasive Pneumococcal Disease, all agesâ€, Confirmed","Invasive Pneumococcal Disease, all agesâ€, Probable", "Meningococcal disease, all serogroups", "Shigellosis", "Syphilis, primary and secondary"]
        droplet_spread = ["Haemophilus influenzae, invasive disease (all ages, all serotypes)", "Invasive Pneumococcal Disease, Age LT 5â€, Confirmed", "Invasive Pneumococcal Disease, Age LT 5â€, Probable", "Invasive Pneumococcal Disease, all agesâ€, Confirmed", "Invasive Pneumococcal Disease, all agesâ€, Probable", "Meningococcal disease, all serogroups", "Mumps", "Pertussis", "Rubella", "Rubella, congenital syndrome"]
        airborne = ["Coccidioidomycosis", "Legionellosis", "Mumps", "Pertussis"]
        vector_borne = ["Babesiosis", "Dengue Virus Infections, Dengueâ€", "Dengue Virus Infections, Severe Dengue", "Ehrlichiosis and Anaplasmosis, Anaplasma phagocytophilum infection", "Ehrlichiosis and Anaplasmosis, Ehrlichia chaffeensis infection", "Ehrlichiosis and Anaplasmosis, Ehrlichia ewingii infection", "Ehrlichiosis and Anaplasmosis, Undetermined Ehrlichiosis/Anaplasmosis", "Malaria", "Rabies, animal", "Spotted Fever Rickettsiosis, Confirmed", "Spotted Fever Rickettsiosis, Probable", "Tetanus", "Varicella morbidity", "West Nile virus diseaseâ€, Neuroinvasive", "West Nile virus diseaseâ€, Nonneuroinvasive", "Zika virus disease, non-congenital"]
        vehicle_borne = ["Campylobacteriosis", "Carbapenemase-producing carbapenem-resistant Enterobacteriaceae â€, Enterobacter spp.", "Carbapenemase-producing carbapenem-resistant Enterobacteriaceae â€, Escherichia coli", "Cryptosporidiosis", "Giardiasis", "Hepatitis (viral, acute, by type)â€, A", "Hepatitis (viral, acute, by type)â€, B", "Hepatitis (viral, acute, by type)â€, C, Confirmed", "Hepatitis (viral, acute, by type)â€, C, Probable", "Salmonellosis (excluding Paratyphoid fever andTyphoid fever)â€", "Shiga toxin-producing Escherichia coli", "Shigellosis", "Syphilis, primary and secondary", "Vibriosis (Any species of the family Vibrionaceae, other than toxigenic Vibrio cholerae O1 or O139), Confirmed", "Vibriosis (Any species of the family Vibrionaceae, other than toxigenic Vibrio cholerae O1 or O139), Probable"]
        categorized_disease = [person_to_person, droplet_spread, airborne, vector_borne, vehicle_borne]

        print("----------Select Category of Disease to Compare-----------")
        test_frame.print_arr(disease_arr)
        print(str(len(disease_arr) + 1) + " All Categories")
        print(str(len(disease_arr) + 2) + " Back to Start")
        print("----------Input-----------")
        print("Enter the number (#) of your choice: ")
        disease_inp = int(input())
        os.system("cls")

        if(disease_inp == 1):
            print(test_frame.linear_reg_table("Person-to-Person Transmission", attribute_arr[attribute_inp], categorized_disease[0], new_attrib_arr[attribute_second_inp]))
            print("----------Would You Like to Save it in a CSV File?-----------")
            print("1 Yes")
            print("2 No. Go Back to Start")
            print("----------Input-----------")
            print("Enter the number (#) of your choice: ")
            save_inp = int(input())
            os.system("cls")

            if(save_inp == 1):
                table_df = test_frame.linear_reg_table("Person-to-Person Transmission", attribute_arr[attribute_inp], categorized_disease[0], new_attrib_arr[attribute_second_inp])
                table_df.to_csv(os.path.dirname(__file__) + "/Person-to-Person Transmission and " + attribute_arr[attribute_inp] + ".csv")
            elif(save_inp == 2):
                os.system("cls")
                continue
        elif(disease_inp == 2):
            print(test_frame.linear_reg_table("Droplet Spread", attribute_arr[attribute_inp], categorized_disease[1], new_attrib_arr[attribute_second_inp]))
            
            print("----------Would You Like to Save it in a CSV File?-----------")
            print("1 Yes")
            print("2 No. Go Back to Start")
            print("----------Input-----------")
            print("Enter the number (#) of your choice: ")
            save_inp = int(input())
            os.system("cls")

            if(save_inp == 1):
                table_df = test_frame.linear_reg_table("Droplet Spread", attribute_arr[attribute_inp], categorized_disease[1], new_attrib_arr[attribute_second_inp])
                table_df.to_csv(os.path.dirname(__file__) + "/Droplet Spread and " + attribute_arr[attribute_inp] + ".csv")
            elif(save_inp == 2):
                os.system("cls")
                continue
        elif(disease_inp == 3):
            print(test_frame.linear_reg_table("Air-borne Transmission", attribute_arr[attribute_inp], categorized_disease[2], new_attrib_arr[attribute_second_inp]))
            
            print("----------Would You Like to Save it in a CSV File?-----------")
            print("1 Yes")
            print("2 No. Go Back to Start")
            print("----------Input-----------")
            print("Enter the number (#) of your choice: ")
            save_inp = int(input())
            os.system("cls")

            if(save_inp == 1):
                table_df = test_frame.linear_reg_table("Air-borne Transmission", attribute_arr[attribute_inp], categorized_disease[2], new_attrib_arr[attribute_second_inp])
                table_df.to_csv(os.path.dirname(__file__) + "/Air-borne Transmission and " + attribute_arr[attribute_inp] + ".csv")
            elif(save_inp == 2):
                os.system("cls")
                continue
        elif(disease_inp == 4):
            print(test_frame.linear_reg_table("Vector-borne Transmission", attribute_arr[attribute_inp], categorized_disease[3], new_attrib_arr[attribute_second_inp]))
            
            print("----------Would You Like to Save it in a CSV File?-----------")
            print("1 Yes")
            print("2 No. Go Back to Start")
            print("----------Input-----------")
            print("Enter the number (#) of your choice: ")
            save_inp = int(input())
            os.system("cls")

            if(save_inp == 1):
                table_df = test_frame.linear_reg_table("Vector-borne Transmission", attribute_arr[attribute_inp], categorized_disease[3], new_attrib_arr[attribute_second_inp])
                table_df.to_csv(os.path.dirname(__file__) + "/Vector-borne Transmission and " + attribute_arr[attribute_inp] + ".csv")
            elif(save_inp == 2):
                os.system("cls")
                continue
        elif(disease_inp == 5):
            print(test_frame.linear_reg_table("Vehicle-borne Transmission", attribute_arr[attribute_inp], categorized_disease[4], new_attrib_arr[attribute_second_inp]))
            
            print("----------Would You Like to Save it in a CSV File?-----------")
            print("1 Yes")
            print("2 No. Go Back to Start")
            print("----------Input-----------")
            print("Enter the number (#) of your choice: ")
            save_inp = int(input())
            os.system("cls")

            if(save_inp == 1):
                table_df = test_frame.linear_reg_table("Vehicle-borne Transmission", attribute_arr[attribute_inp], categorized_disease[4], new_attrib_arr[attribute_second_inp])
                table_df.to_csv(os.path.dirname(__file__) + "/Vehicle-borne Transmission and " + attribute_arr[attribute_inp] + ".csv")
            elif(save_inp == 2):
                os.system("cls")
                continue
        elif(disease_inp == 6):
            print(test_frame.linear_reg_table("Person-to-Person Transmission", attribute_arr[attribute_inp], categorized_disease[0], new_attrib_arr[attribute_second_inp]))
            print()
            print(test_frame.linear_reg_table("Droplet Spread", attribute_arr[attribute_inp], categorized_disease[1], new_attrib_arr[attribute_second_inp]))
            print()
            print(test_frame.linear_reg_table("Air-borne Transmission", attribute_arr[attribute_inp], categorized_disease[2], new_attrib_arr[attribute_second_inp]))
            print()
            print(test_frame.linear_reg_table("Vector-borne Transmission", attribute_arr[attribute_inp], categorized_disease[3], new_attrib_arr[attribute_second_inp]))
            print()
            print(test_frame.linear_reg_table("Vehicle-borne Transmission", attribute_arr[attribute_inp], categorized_disease[4], new_attrib_arr[attribute_second_inp]))
            

            print("----------Would You Like to Save it in a CSV File?-----------")
            print("1 Yes")
            print("2 No. Go Back to Start")
            print("----------Input-----------")
            print("Enter the number (#) of your choice: ")
            save_inp = int(input())
            os.system("cls")

            if(save_inp == 1):
                table_df = test_frame.linear_reg_table("Person-to-Person Transmission", attribute_arr[attribute_inp], categorized_disease[0], new_attrib_arr[attribute_second_inp])
                table_df.to_csv(os.path.dirname(__file__) + "/Person-to-Person Transmission and " + attribute_arr[attribute_inp] + " and " + new_attrib_arr[attribute_second_inp] + ".csv")

                table_df = test_frame.linear_reg_table("Droplet Spread", attribute_arr[attribute_inp], categorized_disease[1], new_attrib_arr[attribute_second_inp])
                table_df.to_csv(os.path.dirname(__file__) + "/Droplet Spread and " + attribute_arr[attribute_inp] + " and " + new_attrib_arr[attribute_second_inp] + ".csv")

                table_df = test_frame.linear_reg_table("Air-borne Transmission", attribute_arr[attribute_inp], categorized_disease[2], new_attrib_arr[attribute_second_inp])
                table_df.to_csv(os.path.dirname(__file__) + "/Air-borne Transmission and " + attribute_arr[attribute_inp] + " and " + new_attrib_arr[attribute_second_inp] + ".csv")

                table_df = test_frame.linear_reg_table("Vector-borne Transmission", attribute_arr[attribute_inp], categorized_disease[3], new_attrib_arr[attribute_second_inp])
                table_df.to_csv(os.path.dirname(__file__) + "/Vector-borne Transmission and " + attribute_arr[attribute_inp] + " and " + new_attrib_arr[attribute_second_inp] + ".csv")

                table_df = test_frame.linear_reg_table("Vehicle-borne Transmission", attribute_arr[attribute_inp], categorized_disease[4], new_attrib_arr[attribute_second_inp])
                table_df.to_csv(os.path.dirname(__file__) + "/Vehicle-borne Transmission and " + attribute_arr[attribute_inp] + " and " + new_attrib_arr[attribute_second_inp] + ".csv")
            elif(save_inp == 2):
                os.system("cls")
                continue
        elif(disease_inp == len(disease_arr) + 2):
            os.system("cls")
            continue

    elif(fig_inp == 8):
        test_frame.polynomial_reg(attribute_arr[attribute_inp], disease_arr[disease_inp])
    elif(fig_inp == 9):
        test_frame.polynomial_reg_summary(attribute_arr[attribute_inp], disease_arr[disease_inp])
    elif(fig_inp == 10):
        test_frame.naive_bayes_cm(attribute_arr, disease_arr[disease_inp], cv_kfold=10, class_bins=4,nb_type='complement',n_features=10) ##standard scaler

    # elif(fig_inp == 11):
    #     #code here for naive-bayes, result only
    elif(fig_inp == 12):
        test_frame.k_means_cc(attribute_arr, disease_arr[disease_inp], 4, min_sil_score=0.60, k_finder=0)
    # elif(fig_inp == 13):
    #     #code here for kmeans, result only

    print("Would you like to try again <Y/N>? ")
    fig_inp = input()
    os.system("cls")

print("User has closed the program.")
