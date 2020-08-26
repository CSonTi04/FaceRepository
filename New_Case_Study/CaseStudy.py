# Databricks notebook source
import numpy as np
import pandas as pd
import seaborn as sns
import time
import re
import os
import pyspark
import matplotlib.pyplot as plt
from pyspark import SparkContext
sns.set(style="darkgrid")

# COMMAND ----------

#df = spark.read.csv('/FileStore/tables/Adatok.csv', sep = ';', header = "True")
df = spark.read.csv('/FileStore/tables/Distances-3.csv', sep = ';', header = "True")
df = df.toPandas()
df.Distance = df.Distance.astype("double")
df.head(5)

# COMMAND ----------

distp = sns.distplot(df['Distance'], hist=True, kde=False,color = 'blue',hist_kws={'edgecolor':'black'})
plt.axvline(1.1, color="red", linestyle="--");

# COMMAND ----------

df_128 = df[df.Dimension == '128']
distp_128 = sns.distplot(df_128['Distance'], hist=True, kde=False, color = 'blue',hist_kws={'edgecolor':'black'})
plt.axvline(1.1, color="red", linestyle="--");

# COMMAND ----------

df_512 = df[df.Dimension == '512']
distp = sns.distplot(df_512['Distance'], hist=True, kde=False, color = 'blue',hist_kws={'edgecolor':'black'})
plt.axvline(1.1, color="red", linestyle="--");

# COMMAND ----------

df_within_t = df[(df.Within_User == '0') & (df.Dimension == '128')]
distp = sns.distplot(df_within_t['Distance'], hist=True, kde=False, color = 'yellow',hist_kws={'edgecolor':'black'})
plt.axvline(1.1, color="red", linestyle="--");

# COMMAND ----------

df_within_f = df[(df.Within_User == '1') & (df.Dimension == '128')]
distp = sns.distplot(df_within_f['Distance'], hist=True, kde=False, color = 'yellow',hist_kws={'edgecolor':'black'})
plt.axvline(1.1, color="red", linestyle="--");

# COMMAND ----------

df_within_f = df[(df.Within_User == '0') & (df.Dimension == '512')]
distp = sns.distplot(df_within_f['Distance'], hist=True, kde=False, color = 'yellow',hist_kws={'edgecolor':'black'})
plt.axvline(1.1, color="red", linestyle="--");

# COMMAND ----------

df_within_f = df[(df.Within_User == '1') & (df.Dimension == '512')]
distp = sns.distplot(df_within_f['Distance'], hist=True, kde=False, color = 'yellow',hist_kws={'edgecolor':'black'})
plt.axvline(1.1, color="red", linestyle="--");

# COMMAND ----------

cat = sns.catplot(x="Ref_User", y="Distance", hue="Dimension", kind="swarm", data=df);
plt.axhline(1.1, color="red", linestyle="--");

# COMMAND ----------

cat = sns.catplot(x="Emotion", y="Distance", hue="Intensity", kind="swarm", data=df, palette=sns.xkcd_palette(["green","red"]));
plt.axhline(1.1, color="red", linestyle="--");

# COMMAND ----------

cat = sns.catplot(x="Ref_User", y="Distance", hue="Emotion", kind="swarm", data=df, palette=sns.xkcd_palette(["amber","dusty purple"]));
plt.axhline(1.1, color="red", linestyle="--");

# COMMAND ----------

df_128 = df[(df.Within_User == '0') & (df.Dimension == '128')]
cat_128 = sns.catplot(x="Ref_User", y="Distance",hue="Intensity", kind="swarm", data=df_128);
plt.axhline(1.1, color="red", linestyle="--");

# COMMAND ----------

df_128 = df[(df.Within_User == '0') & (df.Dimension == '128')]
cat_128 = sns.catplot(x="Ref_User", y="Distance",hue="Emotion", kind="swarm", data=df_128, palette=sns.xkcd_palette(["amber","dusty purple"]));
plt.axhline(1.1, color="red", linestyle="--");

# COMMAND ----------

df_128 = df[(df.Within_User == '0') & (df.Dimension == '128')]
cat = sns.catplot(x="Emotion", y="Distance", hue="Intensity", kind="swarm", data=df_128, palette=sns.xkcd_palette(["green","red"]));
plt.axhline(1.1, color="red", linestyle="--");

# COMMAND ----------

df_128 = df[(df.Within_User == '1') & (df.Dimension == '128')]
cat_128 = sns.catplot(x="Ref_User", y="Distance",hue="Intensity", kind="swarm", data=df_128);
plt.axhline(1.1, color="red", linestyle="--");

# COMMAND ----------

df_128 = df[(df.Within_User == '1') & (df.Dimension == '128')]
cat_128 = sns.catplot(x="Ref_User", y="Distance",hue="Emotion", kind="swarm", data=df_128,palette=sns.xkcd_palette(["amber","dusty purple"]));
plt.axhline(1.1, color="red", linestyle="--");

# COMMAND ----------

df_128 = df[(df.Within_User == '1') & (df.Dimension == '128')]
cat = sns.catplot(x="Emotion", y="Distance", hue="Intensity", kind="swarm", data=df_128, palette=sns.xkcd_palette(["green","red"]));
plt.axhline(1.1, color="red", linestyle="--");

# COMMAND ----------

df_512 = df[(df.Within_User == '0') & (df.Dimension == '512')]
cat_512 = sns.catplot(x="Ref_User", y="Distance",hue="Intensity", kind="swarm", data=df_512);
plt.axhline(1.1, color="red", linestyle="--");

# COMMAND ----------

df_512 = df[(df.Within_User == '0') & (df.Dimension == '512')]
cat_512 = sns.catplot(x="Ref_User", y="Distance",hue="Emotion", kind="swarm", data=df_512,palette=sns.xkcd_palette(["amber","dusty purple"]));
plt.axhline(1.1, color="red", linestyle="--");

# COMMAND ----------

df_512 = df[(df.Within_User == '0') & (df.Dimension == '512')]
cat = sns.catplot(x="Emotion", y="Distance", hue="Intensity", kind="swarm", data=df_512, palette=sns.xkcd_palette(["green","red"]));
plt.axhline(1.1, color="red", linestyle="--");

# COMMAND ----------

df_512 = df[(df.Within_User == '1') & (df.Dimension == '512')]
cat_512 = sns.catplot(x="Ref_User", y="Distance",hue="Intensity", kind="swarm", data=df_512);
plt.axhline(1.1, color="red", linestyle="--");

# COMMAND ----------

df_512 = df[(df.Within_User == '1') & (df.Dimension == '512')]
cat_512 = sns.catplot(x="Ref_User", y="Distance",hue="Emotion", kind="swarm", data=df_512, palette=sns.xkcd_palette(["amber","dusty purple"]));
plt.axhline(1.1, color="red", linestyle="--");

# COMMAND ----------

df_512 = df[(df.Within_User == '1') & (df.Dimension == '512')]
cat = sns.catplot(x="Emotion", y="Distance", hue="Intensity", kind="swarm", data=df_512, palette=sns.xkcd_palette(["green","red"]));
plt.axhline(1.1, color="red", linestyle="--");

# COMMAND ----------

df_512_128 = df
cat = sns.catplot(x="Dimension", y="Distance", hue="Within_User", kind="swarm", data=df_512_128, palette=sns.xkcd_palette(["green","red"]));
plt.axhline(1.1, color="red", linestyle="--");

# COMMAND ----------

df.Ref_User_Emotion_Percentage = df.Ref_User_Emotion_Percentage.astype("int32")
df.Eval_User_Emotion_Percentage = df.Eval_User_Emotion_Percentage.astype("int32")
df.Dimension = df.Dimension.astype("int32")
df.Within_User = df.Within_User.astype("int32")

# COMMAND ----------

corr_df = df.corr()
sns.heatmap(corr_df, xticklabels=corr_df.columns.values, yticklabels=corr_df.columns.values, annot = True, annot_kws={'size':12})
heat_map=plt.gcf(); heat_map.set_size_inches(10,8)
plt.xticks(fontsize=10); plt.yticks(fontsize=10); 
plt.show()

# COMMAND ----------

df.Emotion = df.Emotion.astype("str")
df.Intensity = df.Intensity.astype("str")
df['Emotion_And_Intensity'] = df[['Emotion', 'Intensity']].apply(lambda x: '_'.join(x), axis=1)
df.head(5)

# COMMAND ----------

df_128_0 = df[(df.Within_User == '0') & (df.Dimension == '128')]
cat_128_0 = sns.catplot(x="Ref_User", y="Distance",hue="Emotion_And_Intensity", kind="swarm", data=df_128_0, palette=sns.xkcd_palette(["amber","dusty purple","tea","royal"]));
plt.axhline(1.1, color="red", linestyle="--");

# COMMAND ----------

df_128_1 = df[(df.Within_User == '1') & (df.Dimension == '128')]
cat_128_1 = sns.catplot(x="Ref_User", y="Distance",hue="Emotion_And_Intensity", kind="swarm", data=df_128_1, palette=sns.xkcd_palette(["amber","dusty purple","tea","royal"]));
plt.axhline(1.1, color="red", linestyle="--");

# COMMAND ----------

df_512_0 = df[(df.Within_User == '0') & (df.Dimension == '512')]
cat_512_0 = sns.catplot(x="Ref_User", y="Distance",hue="Emotion_And_Intensity", kind="swarm", data=df_512_0, palette=sns.xkcd_palette(["amber","dusty purple","tea","royal"]));
plt.axhline(1.1, color="red", linestyle="--");

# COMMAND ----------

df_512_1 = df[(df.Within_User == '1') & (df.Dimension == '512')]
cat_512_1 = sns.catplot(x="Ref_User", y="Distance",hue="Emotion_And_Intensity", kind="swarm", data=df_512_1, palette=sns.xkcd_palette(["amber","dusty purple","tea","royal"]));
plt.axhline(1.1, color="red", linestyle="--");

# COMMAND ----------


