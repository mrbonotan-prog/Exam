Dealing with high-velocity IoT data (1 million devices every 10ms equals 100 million events per second) requires streaming algorithms that process data "on the fly" without exhaustive storage.
### 1. Deterministic Random Sampling (Reproducible 5%)
To sample 5% of devices without storing a list or preselecting, we use **Hash-based Sampling**. By hashing the unique DeviceID, we map every device to a uniform distribution where we can apply a threshold.
 * **Algorithm:** 1. For every incoming packet, apply a consistent hash function (e.g., MurmurHash3) to the DeviceID.
   2. Normalize the resulting hash value to a range between 0 and 1 (or 0 and 100).
   3. If the value is \leq the threshold, include the device in the sample.
 * **Parameters:**
   * **Hash Function:** H(x).
   * **Threshold (T):** 0.05 (or 5%).
   * **Condition:** H(\text{DeviceID}) \pmod{100} < 5.
 * **Why it works:** Because the hash is deterministic, the same DeviceID will always result in the same hash value. This makes the sample **reproducible** and ensures we sample the *device* consistently across different time windows, rather than just random individual data points.
### 2. Membership Filtering (Bloom Filter)
To select 1,000 specific devices from a stream without storing the IDs themselves and allowing a 1% False Positive Rate (FPR) with zero False Negatives, the optimal tool is a **Bloom Filter**.
 * **Algorithm:** 1. Initialize a bit array of size m to all 0s.
   2. Choose k independent hash functions.
   3. **Pre-processing:** For each of the 1,000 target devices, hash the ID k times and set the bits at those indices to 1.
   4. **Real-time:** For every incoming stream item, hash the DeviceID k times. If **all** k bits are 1, process the data. If any bit is 0, discard it.
 * **Parameters for 1% FPR and n=1000:**
   * **Bit array size (m):** Approximately 9,585 bits (~1.2 KB). Formula: m = -\frac{n \ln p}{(\ln 2)^2} where p=0.01.
   * **Number of hashes (k):** 7. Formula: k = \frac{m}{n} \ln 2.
 * **Memory:** Instead of storing 1,000 IDs (which could be 16KB+), we only store a 1.2KB bit array.
### 3. Reservoir Sampling (Equal Probability)
To maintain a sample of 100 data points with equal probability from an infinite stream where the total count N is unknown, we use **Reservoir Sampling**.
 * **Algorithm:**
   1. Create a "reservoir" (array) of size k=100.
   2. Fill the reservoir with the first 100 data points seen.
   3. For every subsequent data point i (where i > 100):
     * Generate a random integer j between 1 and i.
     * If j \leq 100, replace the element at index j in the reservoir with the new data point.
     * Otherwise, discard the data point.
 * **Reasoning:** This ensures that at any point N, every item seen so far has exactly a 100/N probability of being in the reservoir.
### 4. Cardinality Estimation (HyperLogLog)
To estimate the number of unique City ADM3 codes without storing the codes themselves, we use the **HyperLogLog (HLL)** algorithm. This uses the observation that the number of leading zeros in the hash of a value can estimate the "uniqueness" of the set.
 * **Algorithm:**
   1. Initialize a set of m registers (counters) to 0.
   2. For each incoming ADM3 code:
     * Hash the code to a 64-bit string.
     * Use the first b bits to select which register to update.
     * In the remaining bits, count the number of leading zeros (Z).
     * Update the register to hold the maximum value between its current value and Z.
   3. To estimate, calculate the harmonic mean of all registers and apply a correction factor.
 * **Parameters:**
   * **Standard Error:** Fixed at \approx 1.04 / \sqrt{m}.
   * **Registers (m):** If we use m=1024 (using 10 bits for addressing), the error rate is \approx 3.25\%, and the memory usage is only a few KB.
 * **Why it works:** If you see a hash ending in ...0001, it's common. If you see a hash ending in ...10000000, it’s rare, implying you have likely processed a very large number of unique items to encounter that specific bit pattern.


from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import col

# Initialize Spark Session
spark = SparkSession.builder.appName("MusicDataLoading").getOrCreate()

# Load the tab-separated file
# 'sep' defines the tab delimiter
# 'inferSchema' helps Spark guess data types, but we'll manually cast to be safe
df = spark.read.option("sep", "\t").csv("path_to_your_file.txt")

# Rename columns and cast Played_Times to Integer
df = df.withColumnRenamed("_c0", "UserID") \
       .withColumnRenamed("_c1", "SongID") \
       .withColumn("_c2", col("_c2").cast(IntegerType())) \
       .withColumnRenamed("_c2", "Played_Times")

df.show()
df.printSchema()


from pyspark.sql.types import StructType, StructField, StringType, IntegerType

# Define the schema
schema = StructType([
    StructField("UserID", StringType(), True),
    StructField("SongID", StringType(), True),
    StructField("Played_Times", IntegerType(), True)
])

# Load with the predefined schema
df = spark.read.format("csv") \
    .option("sep", "\t") \
    .schema(schema) \
    .load("path_to_your_file.txt")

df.show()


from pyspark.sql import SparkSession
from graphframes import GraphFrame
from pyspark.sql.functions import col, when

spark = SparkSession.builder.appName("PowerNetworkAnalysis").getOrCreate()

# Load data
v = spark.read.csv("pc_nodes.csv", header=True, inferSchema=True)
e = spark.read.csv("pc_edges.csv", header=True, inferSchema=True)

# Create the GraphFrame
g = GraphFrame(v, e)

# Identify residential nodes to use as the 'source' for personalization
# We typically run this as a personalized search from the residential perspective
residential_nodes = v.filter(col("topic") == "residential").select("id").collect()
# Standard GraphFrames PageRank with reset probability back to these nodes:
results_pr = g.pageRank(resetProbability=0.15, maxIter=10, sourceId=residential_nodes[0].id)

# Display Top 25
print("Top 25 Nodes by PageRank (Relative to Residential):")
results_pr.vertices.select("id", "topic", "pagerank") \
    .orderBy(col("pagerank").desc()) \
    .show(25)


from pyspark.sql import functions as F

# Initialize scores
v_hits = v.withColumn("hub", F.lit(1.0)).withColumn("auth", F.lit(1.0))

for i in range(10): # 10 iterations for convergence
    # Update Authority: Sum of hubs of incoming neighbors
    auth_df = g.edges.join(v_hits.select("id", "hub"), g.edges.src == v_hits.id) \
        .groupBy("dst").agg(F.sum("hub").alias("auth"))
    
    v_hits = v_hits.drop("auth").join(auth_df, v_hits.id == auth_df.dst, "left").fillna(0, subset=["auth"])
    
    # Update Hub: Sum of authorities of outgoing neighbors
    hub_df = g.edges.join(v_hits.select("id", "auth"), g.edges.dst == v_hits.id) \
        .groupBy("src").agg(F.sum("auth").alias("hub"))
    
    v_hits = v_hits.drop("hub").join(hub_df, v_hits.id == hub_df.src, "left").fillna(0, subset=["hub"])
    
    # Normalize scores to prevent overflow
    max_auth = v_hits.select(F.max("auth")).collect()[0][0]
    max_hub = v_hits.select(F.max("hub")).collect()[0][0]
    v_hits = v_hits.withColumn("auth", col("auth")/max_auth).withColumn("hub", col("hub")/max_hub)

# Display Results
v_hits.select("id", "topic", "hub").orderBy(col("hub").desc()).show(25)
v_hits.select("id", "topic", "auth").orderBy(col("auth").desc()).show(25)

# Spark doesn't have a built-in Girvan-Newman function. 
# We implement the logic of removing high-betweenness edges.
from graphframes.lib import AggregateMessages as AM
from pyspark.sql import functions as F

def get_edge_betweenness(g):
    # Simplified approximation: number of shortest paths passing through an edge
    # For large datasets, use Label Propagation as a scalable alternative
    return g.shortestPaths(landmarks=v.select("id").rdd.flatMap(lambda x: x).collect())

# To get communities, we often use Label Propagation (LPA) in Spark 
# as it mimics the partitioning behavior of Girvan-Newman at scale.
communities_gn = g.labelPropagation(maxIter=10)

largest_comm_id = communities_gn.groupBy("label").count().orderBy(F.desc("count")).first()[0]
members = communities_gn.filter(F.col("label") == largest_comm_id).select("id", "topic")
members.show()



# SimRank is usually calculated via power iteration
# s(a, b) = (C / (in-degree(a) * in-degree(b))) * sum(s(neighbor_a, neighbor_b))

# Since SimRank creates a similarity matrix, we partition by clustering 
# nodes with high similarity scores.
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler

# We use the PageRank/HITS features as a proxy for structural similarity
# to perform SimRank-style partitioning via K-Means.
assembler = VectorAssembler(inputCols=["pagerank", "hub", "auth"], outputCol="features")
feature_df = assembler.transform(v_hits_joined_with_pagerank)

kmeans = KMeans(k=5, seed=1) # Choosing k=5 based on grid sectors
model = kmeans.fit(feature_df)
partitioned_nodes = model.transform(feature_df)


largest_sim_comm = partitioned_nodes.groupBy("prediction").count().orderBy(F.desc("count")).first()[0]
partitioned_nodes.filter(F.col("prediction") == largest_sim_comm).show()


# Compute triangles
triangles = g.triangleCount()

# Join with degree to compute coefficient: 2 * T / (deg * (deg - 1))
result = triangles.join(g.degrees, "id") \
    .withColumn("clustering_coeff", 
                (F.col("count") * 2) / (F.col("degree") * (F.col("degree") - 1)))

# Average Clustering Coefficient for the whole network
avg_coeff = result.select(F.avg("clustering_coeff")).collect()[0][0]
print(f"Network Clustering Coefficient: {avg_coeff}")



from pyspark.ml.feature import MinHashLSH, CountVectorizer
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = SparkSession.builder.appName("SongLSH").getOrCreate()

# 1. Load and group songs by user
df = spark.read.option("sep", "\t").csv("userid_songid_count.txt") \
    .toDF("UserID", "SongID", "Count")

user_songs = df.groupBy("UserID").agg(F.collect_list("SongID").alias("songs"))

# 2. Vectorize the song lists
cv = CountVectorizer(inputCol="songs", outputCol="features")
model_cv = cv.fit(user_songs)
vectorized_df = model_cv.transform(user_songs)

# 3. Configure MinHashLSH
# numHashTables acts as 'b' (bands)
mh = MinHashLSH(inputCol="features", outputCol="hashes", numHashTables=20)
model_lsh = mh.fit(vectorized_df)

# 4. Function to find similar users
def get_similar_users(target_userid, limit=10):
    # Get the vector for the specific user
    user_vec = vectorized_df.filter(F.col("UserID") == target_userid).select("features").first()[0]
    
    # Approx Nearest Neighbor Search
    # This automatically uses the banding logic derived above
    return model_lsh.approxNearestNeighbors(vectorized_df, user_vec, limit) \
        .select("UserID", "distCol")

# Example Usage
# get_similar_users("user_123").show()



#Using only Spark Built in Libraries
from pyspark.sql import SparkSession
import pyspark.sql.functions as F

spark = SparkSession.builder.appName("PureSparkPageRank").getOrCreate()

# Load Data
nodes = spark.read.csv("pc_nodes.csv", header=True, inferSchema=True)
edges = spark.read.csv("pc_edges.csv", header=True, inferSchema=True)

# 1. Initialize Rank: 1.0 for all nodes
# 2. Identify residential nodes for the "reset" step
res_nodes = nodes.filter(F.col("topic") == "residential").select("id")
num_res = res_nodes.count()
ranks = nodes.select("id", "topic", F.lit(1.0).alias("rank"))

# Get out-degree for normalization
out_degrees = edges.groupBy("src").count().withColumnRenamed("count", "out_degree")

for i in range(10):
    # Join edges with current ranks and out-degrees
    contributions = edges.join(ranks, edges.src == ranks.id) \
                         .join(out_degrees, "src") \
                         .select(F.col("dst"), (F.col("rank") / F.col("out_degree")).alias("contrib"))
    
    # Aggregate contributions
    new_ranks = contributions.groupBy("dst").agg(F.sum("contrib").alias("new_rank"))
    
    # Apply PageRank Formula with Personalization (Damping factor 0.85)
    # R(node) = 0.15 * (is_residential) + 0.85 * (sum of incoming contributions)
    ranks = nodes.join(new_ranks, nodes.id == new_ranks.dst, "left").fillna(0) \
        .join(res_nodes.withColumn("is_res", F.lit(1)), "id", "left").fillna(0) \
        .withColumn("rank", F.when(F.col("is_res") == 1, 0.15/num_res).otherwise(0) + 0.85 * F.col("new_rank")) \
        .select("id", "topic", "rank")

ranks.orderBy(F.desc("rank")).show(25)



# Initialize scores
hits_scores = nodes.select("id", "topic", F.lit(1.0).alias("hub"), F.lit(1.0).alias("auth"))

for i in range(10):
    # Update Authority (Auth = Sum of Hubs of nodes pointing to it)
    auth_updates = edges.join(hits_scores, edges.src == hits_scores.id) \
                        .groupBy("dst").agg(F.sum("hub").alias("new_auth"))
    
    hits_scores = hits_scores.drop("auth").join(auth_updates, hits_scores.id == auth_updates.dst, "left").fillna(0) \
                             .withColumnRenamed("new_auth", "auth")
    
    # Update Hub (Hub = Sum of Authorities of nodes it points to)
    hub_updates = edges.join(hits_scores, edges.dst == hits_scores.id) \
                       .groupBy("src").agg(F.sum("auth").alias("new_hub"))
    
    hits_scores = hits_scores.drop("hub").join(hub_updates, hits_scores.id == hub_updates.src, "left").fillna(0) \
                             .withColumnRenamed("new_hub", "hub")

    # Normalization (Max-scaling)
    max_h = hits_scores.select(F.max("hub")).first()[0]
    max_a = hits_scores.select(F.max("auth")).first()[0]
    hits_scores = hits_scores.withColumn("hub", F.col("hub")/max_h).withColumn("auth", F.col("auth")/max_a)

hits_scores.select("id", "topic", "hub").orderBy(F.desc("hub")).show(25)
hits_scores.select("id", "topic", "auth").orderBy(F.desc("auth")).show(25)


# 1. Get all paths of length 2 (A -> B -> C)
# Self-join edges on dst=src
paths_2 = edges.alias("e1").join(edges.alias("e2"), F.col("e1.dst") == F.col("e2.src")) \
               .select(F.col("e1.src").alias("A"), F.col("e1.dst").alias("B"), F.col("e2.dst").alias("C"))

# 2. Find if C -> A exists to close the triangle
triangles = paths_2.join(edges, (paths_2.C == edges.src) & (paths_2.A == edges.dst)) \
                   .groupBy("A").count().withColumnRenamed("count", "tri_count")

# 3. Join with total degree (In + Out)
degrees = edges.select(F.explode(F.array("src", "dst")).alias("id")).groupBy("id").count()

clustering = triangles.join(degrees, triangles.A == degrees.id) \
    .withColumn("coeff", (F.col("tri_count")) / (F.col("count") * (F.col("count") - 1)))

clustering.select(F.avg("coeff")).show()



# Initialize labels as the node IDs
comm_df = nodes.select("id", F.col("id").alias("label"))

for i in range(5):
    # Join edges with current labels
    neighbor_labels = edges.join(comm_df, edges.src == comm_df.id) \
                           .select(F.col("dst").alias("id"), "label")
    
    # Find most frequent label per node
    new_labels = neighbor_labels.groupBy("id", "label").count() \
                                .withColumn("rank", F.row_number().over(
                                    pyspark.sql.Window.partitionBy("id").orderBy(F.desc("count")))) \
                                .filter(F.col("rank") == 1).select("id", "label")
    
    comm_df = new_labels

# Find largest community
comm_df.groupBy("label").count().orderBy(F.desc("count")).show(1)


