[https://www.mlflow.org/docs/latest/tracking.html]
[https://towardsdatascience.com/experiment-tracking-with-mlflow-in-10-minutes-f7c2128b8f2c]


W notatniku możemy napisać podstawowe polecenia, które miałyby pomóc nam logować i zapisywać informacje dotyczące eksperymentów.

Podstawowe słownictwo:

Experiment  - to zbiór runów

Run - pojedyncze odpalenie kodu na mlflow

Tag - słowa/wyrażenia kluczowe np. informacje o run, cel modelu itp. 

Metrics - miary, które przetrzymujemy w sekcji metryk, muszą być typu numerycznego  (np. miara R2 w regresji liniowej)

Parameters - wartości różne, które możemy zapisywać i przetrzymywać w kodzie run 

Artifacts- pliki, dane moduły, które chcemy przetrzymywać przy odpaleniu kodu

 

Zakładanie eksperymentu:

mlflow.create_experiment(name: str, artifact_location: Optional[str] = None, tags: Optional[Dict[str, Any]] = None )  - tworzenie nowego eksperymentu, gdzie name jest nazwą naszego eksperymentu i jest wymagane, a artifact_location to lokalizacja gdzie będą zapisywać się dane o eksperymencie/modelu i inne informacje (jest opcjonalne), tags są również opcjonalne..

Lokalizacja artifacts musi byc dodana w formie dbfs:/ i  jeżeli chcemy zapisywać np. tabelki, wykresy czy inne rzeczy przez np. log_artifact to należy podać ścieżkę gdzie.

experiment = mlflow.get_experiment(experiment_id) - wskazujemy, na którym eksperymencie będziemy pracować 

Nawet jeżeli samodzielnie nie wskażemy tagsto nasz experiment dostanie z automatu przypisane taki podstawowe, aby sprawdzić wystarczy wskazać print("Tags: {}".format(experiment.tags))


```
import mlflow
import mlflow.pyfunc

experiment_id = mlflow.create_experiment(
    "/Users/izabela.karwowska@ccc.eu/MROI/MROI_experiments",
    artifact_location='dbfs:/FileStore/mlflow/mroi/MROI_EXPERIMENTS',
   )

experiment = mlflow.get_experiment(experiment_id)
print("Name: {}".format(experiment.name))
print("Experiment_id: {}".format(experiment.experiment_id))
print("Artifact Location: {}".format(experiment.artifact_location))
print("Tags: {}".format(experiment.tags))
print("Lifecycle_stage: {}".format(experiment.lifecycle_stage))
```

Jeżeli stworzymy nasz experiment  to mamy już nasz zbiór gdzie możemy odpalać pierwsze runy. Możemy pracować na kilku eksperymentach. Ważne jest, ze przy każdym założeniu nowego eksperymentu z automatu nadawany jest experiment_id, którym powinniśmy się później posługiwać by wskazać o jaki zbiór nam chodzi. Przykładowo możemy we dwie osoby pracować na jednym eksperymencie, ale każdy może mieć również swój własny experiment i tu jest ważny wtedy experiment_id. 

Kod do odpalenia Run

```
with mlflow.start_run(experiment_id='2919680012018678') as run:
  mlflow.log_param
  mlflow.log_metric
  mlflow.log_artifact 
  mlflow.log_figure
```

```
with mlflow.start_run(experiment_id='2919680012018678'):
        
        #Etap modelowania  
        correlation_table = pd.DataFrame(data_df.corr()) 
        
        date = X['date']
        X = X.drop('date',axis=1)
        X = sm.add_constant(X)  
        y_values = y
        print(f'Wielkość zbioru danych objaśniających:', X.shape)
        print(f'Wielkość zmiennej ojaśnianej:', y.shape)
        
        model = sm.OLS(y,X).fit() #tworzenie modelu
        predicted_qualities = model.fittedvalues #wartości estymowane
        residuals = model.resid #wartości błędu
        list_of_variables = model.params #lista zmiennych
        X_list = X.columns.to_list() #nazwy zmiennych
        print(data_df.columns.to_list())
        alpha = pd.DataFrame(list_of_variables).filter(like = 'const', axis=0).iloc[0] #wartość stała
        list_of_betas = pd.DataFrame(list_of_variables).filter(items = X_list, axis=0) #wartości zmiennych 
        
        #Tabela zbierająca wartości estymowane, rzeczywiste i błąd
        actual_predicted = pd.DataFrame({'date':date,  
                                         'Modelled sales': predicted_qualities, 
                                         'Residuals': residuals})
        actual_predicted = pd.merge(actual_predicted, y_values, left_index=True, right_index=True)
        actual_predicted.columns=['date','Modelled sales','Residuals','Actual sales']
        print(actual_predicted.head())
        model_sum = model.summary() #podsumowanie modelu na wydruku
        print(model_sum)        
        
        #Wyliczenie najważniejszych metryk, statystyk i testów
        (rmse, mse, mae, r2) = eval_metrics(y, predicted_qualities)
        (positive_items,positive_values,negative_items,negative_values) = check_feature_values(list_of_betas)
        (check_features_qlt) = check_features_number(positive_items, negative_items)
        (breuschpagan_statistic, breuschpagan_statistic_pvalue, breuschpagan_F_statistic, breuschpagan_F_statistic_pvalue,check) = heteroscedasticity_het_breuschpagan_test(model)
        (Jarque_Bera, JBpv, Skew, Kurtosis,JB_interpret, Skew_interpret) = normality_tests(model)
        (durbin_watson_test, durbin_watson_interpret) = autocorrelation_test(model.resid)
        
        # Wypisanie wszystkich metryk
        print("intercept: %s" % round(alpha,2))
        print("RMSE: %s" % round(rmse,2))
        print("MSE: %s" % round(mse,2))
        print("MAE: %s" % round(mae,2))
        print("R2: %s" % round(r2,2))
        #print("Jarque_Bera %s" % round(Jarque_Bera,2))
        #print("Jarque_Bera_pv %s" % round(JBpv,2))
        #print("Skew %s" % round(Skew,2))
        #print("Kurtosis %s" % round(Kurtosis,2))
        #print("durbin_watson_test %s" % round(durbin_watson_test,2))
        
        #Stworzenie wykresu liniowego dla wartości estymowanej i rzeczywistej
        actual_predicted_main = actual_vs_predicted_graph(actual_predicted, ['Modelled sales', 'Actual sales', 'Residuals'], 
                                                  save = False, colors = colors)
        actual_predicted_main.show()
        
        #Stworzenie wykresu dekompozycji sprzedaży 
        base_variables = ['const']
        
        decomp_df = decomposition(data_df_date.drop(['turnover'],axis=1), model, base_vars = base_variables, 
                                                  #min_vars = [], #variables whose minimum goes to the base
                                                  #max_vars = [], #variables whose maximum goes to the base
                                                  decomp_type = 'add') #decomp_type = 'add'/'log'
        
        #Wykres dekompozycji w czasie 
        decomposition_area_chart = decomposition_area_plot(decomp_df, colors = colors, save = False)
        decomposition_area_chart.show()
        
        #Wykres dekompozycji 
       
        decomposition_bar_chart_proc_yearly = decomposition_bar_plot(decomp_df, yearly_proc=True, 
                                                      colors = colors, save = False,   
                                                      file_name = 'decomposition_bar_plot_proc', 
                                                      y_title = "<b>Revenue (%)</b>")
        decomposition_bar_chart_proc_yearly.show()
        
        #Zapisanie ID aktualnego testu modelu
        runID = mlflow.active_run().info.run_id
        
        #Zapisywanie wyników z MLFlow
        # Logowanie metryk w głównym widoku eksperymentów
        mlflow.log_metric("rmse", round(rmse,2))
        mlflow.log_metric("mse", round(mse,2))
        mlflow.log_metric("r2", round(r2,2))
        mlflow.log_metric("mae", round(mae,2))
        mlflow.log_metric("intercept", round(alpha,2))
                
        # Logowanie różnych parametrów
        mlflow.log_param('run_id', runID)
        mlflow.log_param('1_model_variables_count',len(positive_items)+len(negative_items)) # ilość zmiennych
        mlflow.log_param('2_positive_variables_count',len(positive_items)) # ilość zmiennych z dodatnim wpływem
        mlflow.log_param('3_negative_variables_count',len(negative_items)) #ilość zmiennych z ujemnym wpływem
        mlflow.log_param('4_More than 16 variables',check_features_qlt) #czy więcej niż 16 zmiennych
        mlflow.log_param('5_heteroscedasticity_breuschpagan_statistic',breuschpagan_statistic)
        mlflow.log_param('5_heteroscedasticity_breuschpagan_statistic_pvalue',breuschpagan_statistic_pvalue)
        mlflow.log_param('5_heteroscedasticity_breuschpagan_F_statistic',breuschpagan_F_statistic)
        mlflow.log_param('5_heteroscedasticity_breuschpagan_F_statistic_pvalue',breuschpagan_F_statistic_pvalue)
        mlflow.log_param('5_heteroscedasticity_breuschpagan_F_statistic_pvalue',breuschpagan_F_statistic_pvalue)
        mlflow.log_param('5_heteroscedasticity_pvalue_check',check)
        mlflow.log_param('6_Jarque_Bera_normality_test',Jarque_Bera)
        mlflow.log_param('6_Jarque_Bera_pvalue',JBpv)
        mlflow.log_param('6_Jarque_Bera_check',JB_interpret)
        mlflow.log_param('6_Skew_normality_test',Skew)
        mlflow.log_param('6_Skew_normality_check',Skew_interpret)
        mlflow.log_param('6_Kurtosis_normality_test',Kurtosis)
        mlflow.log_param('7_Durbin_watson_autokorelation_test',durbin_watson_test)
        mlflow.log_param('7_Durbin_watson_autokorelation_check',durbin_watson_interpret)        
        
        #Zapisywanie rzeczy do folderów w artifaktach
        
        #Zapisanie listy zmiennych
        #zapisanie csv
        list_of_variables.to_csv('list_of_variables.csv')
        mlflow.log_artifact('list_of_variables.csv') 
        
        #Zapisanie tabelki jako html
        list_of_variables_save = pd.DataFrame(list_of_variables)
        list_of_variables_save = list_of_variables_save.reset_index()
        list_of_variables_save.columns = ['variable','Beta']
        list_of_variables_save['Beta'] = list_of_variables_save['Beta'].round(2)
        list_of_variables_save.to_html("list_of_variables.html")
        mlflow.log_artifact("list_of_variables.html")
        
        #Zapisanie tabelki korelacji zmiennych
        #Zapisanie do csv
        correlation_table.to_csv('correlation_table.csv')
        mlflow.log_artifact('correlation_table.csv')
        #Zzapisanie jako html
        correlation_table_save = pd.DataFrame(correlation_table)
        correlation_table_save.to_html("correlation_table_save.html")
        mlflow.log_artifact("correlation_table_save.html")
        
        #Zapisanie wydruku podsumowania modelu jako html
        #mlflow.log_figure(model_sum,'model_sum.png')
        model_summary = model_sum.as_csv()
        # mlflow.log_artifact('model_summary.csv')

        #Zapisanie wykresów jako .png
        mlflow.log_figure(decomposition_area_chart, 'decomposition_area_chart.png') #dekompozycja w czasie
        mlflow.log_figure(decomposition_bar_chart_proc_yearly,'decomposition_bar_chart_proc_yearly.png') #dekompozycja %
        mlflow.log_figure(actual_predicted_main,'actual_predicted_line_check.png') #porównanie wartośc estymowana i rzeczywista
        
        #Zapisanie wykresów jako html
        mlflow.log_figure(actual_predicted_main, "actual_predicted_main.html")
        mlflow.log_figure(decomposition_area_chart, "decomposition_area_chart.html") #dekompozycja w czasie
        mlflow.log_figure(decomposition_bar_chart_proc_yearly, "decomposition_bar_chart_proc_yearly.html") #dekompozycja %
        mlflow.log_figure(actual_predicted_main, "actual_predicted_line_check.html") #porównanie wartośc estymowana i rzeczywista
       
        #Logowanie i zapisanie modelu
        mlflow.sklearn.log_model(model, "model")
        #mlflow.sklearn.save_model(model)
        
        ##Stworzenie tabelki do zapisywania podstawowych wyników z możliwością oznaczenia, który model warto analizować wraz z komentarzem
        row = {'run_id': runID, "variable_list": X_list, "r2": r2}
        dataframe_runid =  pd.read_csv('/dbfs/FileStore/mlflow/mroi/' + 'dataframe_runid.csv')
        
        #dataframe_runid = dataframe_runid.drop(['Unnamed: 0'],axis=1)
        dataframe_runid = dataframe_runid.append(row, ignore_index=True,  )
        #dataframe_runid = dataframe_runid.drop(['Unnamed: 0'],axis=1)
     
        dataframe_runid.to_csv('/dbfs/FileStore/mlflow/mroi/' + 'dataframe_runid.csv',index=False)

```
