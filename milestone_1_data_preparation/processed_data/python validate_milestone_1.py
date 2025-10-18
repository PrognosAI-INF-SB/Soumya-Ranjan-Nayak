{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "952450d9-d510-4643-a7b3-2967da6030bc",
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "os.chdir(\"C:/Users/soumy/OneDrive/Desktop/AI-PrognosAI/milestone_1_data_preparation\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "b21f5879-5dcf-4742-80b5-5244dc03e5eb",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "   engine_id  cycle  op_setting_1  op_setting_2  op_setting_3  sensor_1  \\\n",
      "0          1      1       -0.0007       -0.0004         100.0       0.0   \n",
      "1          1      2        0.0019       -0.0003         100.0       0.0   \n",
      "2          1      3       -0.0043        0.0003         100.0       0.0   \n",
      "3          1      4        0.0007        0.0000         100.0       0.0   \n",
      "4          1      5       -0.0019       -0.0002         100.0       0.0   \n",
      "\n",
      "   sensor_2  sensor_3  sensor_4  sensor_5  ...  sensor_13  sensor_14  \\\n",
      "0  0.183735  0.406802  0.309757       0.0  ...   0.205882   0.199608   \n",
      "1  0.283133  0.453019  0.352633       0.0  ...   0.279412   0.162813   \n",
      "2  0.343373  0.369523  0.370527       0.0  ...   0.220588   0.171793   \n",
      "3  0.343373  0.256159  0.331195       0.0  ...   0.294118   0.174889   \n",
      "4  0.349398  0.257467  0.404625       0.0  ...   0.235294   0.174734   \n",
      "\n",
      "   sensor_15  sensor_16  sensor_17  sensor_18  sensor_19  sensor_20  \\\n",
      "0   0.363986        0.0   0.333333        0.0        0.0   0.713178   \n",
      "1   0.411312        0.0   0.333333        0.0        0.0   0.666667   \n",
      "2   0.357445        0.0   0.166667        0.0        0.0   0.627907   \n",
      "3   0.166603        0.0   0.333333        0.0        0.0   0.573643   \n",
      "4   0.402078        0.0   0.416667        0.0        0.0   0.589147   \n",
      "\n",
      "   sensor_21  RUL  \n",
      "0   0.724662  191  \n",
      "1   0.731014  190  \n",
      "2   0.621375  189  \n",
      "3   0.662386  188  \n",
      "4   0.704502  187  \n",
      "\n",
      "[5 rows x 27 columns]\n",
      "   engine_id  cycle  op_setting_1  op_setting_2  op_setting_3  sensor_1  \\\n",
      "0          1      1        0.0023        0.0003         100.0       0.0   \n",
      "1          1      2       -0.0027       -0.0003         100.0       0.0   \n",
      "2          1      3        0.0003        0.0001         100.0       0.0   \n",
      "3          1      4        0.0042        0.0000         100.0       0.0   \n",
      "4          1      5        0.0014        0.0000         100.0       0.0   \n",
      "\n",
      "   sensor_2  sensor_3  sensor_4  sensor_5  ...  sensor_12  sensor_13  \\\n",
      "0  0.545181  0.310661  0.269413       0.0  ...   0.646055   0.220588   \n",
      "1  0.150602  0.379551  0.222316       0.0  ...   0.739872   0.264706   \n",
      "2  0.376506  0.346632  0.322248       0.0  ...   0.699360   0.220588   \n",
      "3  0.370482  0.285154  0.408001       0.0  ...   0.573561   0.250000   \n",
      "4  0.391566  0.352082  0.332039       0.0  ...   0.737740   0.220588   \n",
      "\n",
      "   sensor_14  sensor_15  sensor_16  sensor_17  sensor_18  sensor_19  \\\n",
      "0   0.132160   0.308965        0.0   0.333333        0.0        0.0   \n",
      "1   0.204768   0.213159        0.0   0.416667        0.0        0.0   \n",
      "2   0.155640   0.458638        0.0   0.416667        0.0        0.0   \n",
      "3   0.170090   0.257022        0.0   0.250000        0.0        0.0   \n",
      "4   0.152751   0.300885        0.0   0.166667        0.0        0.0   \n",
      "\n",
      "   sensor_20  sensor_21  \n",
      "0   0.558140   0.661834  \n",
      "1   0.682171   0.686827  \n",
      "2   0.728682   0.721348  \n",
      "3   0.666667   0.662110  \n",
      "4   0.658915   0.716377  \n",
      "\n",
      "[5 rows x 26 columns]\n",
      "   RUL\n",
      "0  112\n",
      "1   98\n",
      "2   69\n",
      "3   82\n",
      "4   91\n",
      "0 0\n"
     ]
    }
   ],
   "source": [
    "import pandas as pd\n",
    "train = pd.read_csv(\"processed_data/train_processed_FD001.csv\")\n",
    "test = pd.read_csv(\"processed_data/test_processed_FD001.csv\")\n",
    "rul = pd.read_csv(\"processed_data/rul_targets_FD001.csv\")\n",
    "\n",
    "print(train.head())\n",
    "print(test.head())\n",
    "print(rul.head())\n",
    "\n",
    "print(train.isnull().sum().sum(), test.isnull().sum().sum())\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "647708b9-e1b7-4493-8b53-47aff062bac6",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
