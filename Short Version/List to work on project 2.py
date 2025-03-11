
# List to work on project 2

# General info for data preprocessing
-	from sklearn.decomposition import PCA # too many columns 
-	le = LabelEncoder()
-	from sklearn.datasets import make_blobs # create fake data, syntehtic data
-	from sklearn.preprocessing import OneHotEncoder, OrdinalEncode
-	from sklearn.compose import ColumnTransformer
-	from sklearn.pipeline import Pipeline
-	StandardScaler().fit_transform(ccinfo_df[["limit_bal"]
-	scaler = MinMaxScaler()
-   correlation matrix

# Time series
-	Time series
-	Prophet 

# Unsupervised learning
-	def encodeMethod():
-	inertia = []
# Create a a list to store the values of k
k = list(range(1, 11))
-	Review the DataFrame : df_elbow
-	import KMeans - model = KMeans
-	Import AgglomerativeClustering
-	Import Birch
-   "Kal"



    # Regression - supervised
-	model = LinearRegression()
-	model = Ridge & Lasso
-	from sklearn.metrics import mean_squared_error, r2_score
-	import train_test_split

    # supervised classification
-	from sklearn.linear_model import LogisticRegression
-	from sklearn.neighbors import KNeighborsClassifier
-	model = SVC
-	model = tree.DecisionTreeClassifier()
-	from sklearn.ensemble import RandomForestClassifier
-	from sklearn.ensemble import ExtraTreesClassifier
-	from sklearn.ensemble import GradientBoostingClassifier
-	from sklearn.ensemble import AdaBoostClassifier
-	from sklearn.ensemble import BaggingClassifier

# classification metrics
-	confusion_matrix
-	classification_repor #
-	roc_auc_score
-	balanced_accuracy_score

# feature engineering 
-	from sklearn.model_selection import GridSearchCV # searching for right level of depth

-	from sklearn.model_selection import RandomizedSearchCV

-	from imblearn.under_sampling import RandomUnderSampler #data not balanced
-	from imblearn.over_sampling import RandomOverSampler #data not balanced
-	from imblearn.under_sampling import ClusterCentroids #data not balanced
-	from imblearn.over_sampling import SMOTE
-	from imblearn.combine import SMOTEENN




M13 Class3 - 01
OneHotEncoder
LabelEncoder()