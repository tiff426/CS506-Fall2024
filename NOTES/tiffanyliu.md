### 9/16/24
* how to collab
  * first fork the repo (essentially creates your own copy, but you gte a copy of the repo at this time)
  * clone your forked repo to your ipan (git clone <...>)
  * create remote called origin pointing to your repo
      * git remote add origin ( + forked repo link)
  * create remote called upstream pointing to the original repo
      * git remote add upstream ( + og repo link)
  * then work in your repo, add, commit
  * then make a PR to the original repo to request that your changes be reviewed and approved 
* git pull upstream main to get the changes from the og repo onto your forked repo
* git push origin main to put your changes (on you local like in your vs code) onto your forked repo (remote, like on github.com)
* whenever you develop you should be in your forked repo
  * be in main to sync the repos
  * BUT THEN when you start to actually develop cut a branch off of main!!! 
* DATA SCIENCE IS REALY REALLY HARD!
* challenges of DS
  * a set of examples may not always be representative of the underlying rule
    * like think about the triples, jsut beause you know (2, 4, 6) works, you may not be able to tell what the rule is
  * there may be infinitely many rules that match example
  * rules/examples may change over time
* all models are wrong but some are useful
* if we have a hypothesis
  * positive example: an example that follows the hypothesis
  * negative example: an example that doesn't follow hypothesis
* rule != hypothesis
  * hypothesis is literally jsut what you thinnk rule is
  * ( ( ( your hypothesis ) rule ) all possible examples ) -> sorry idk how to format
* confirmation bias: looking for and processing information that already fits in with your beliefs/hypothesis
* DS workflow
  * process data -> explore data -> extract features -> create model
    * but at any one of these step, your next option could go back to one of the previous steps, this is not a rigid workflow
    * features: what do you think impacts the outcome
* types of data
  * record: m-dimensional vector
    * i.e. (name, age, balance) -> ("john", 20, 100)
  * graphs -> adjacency matrix
  
  
### 9/18/24
* clusters can be ambiguous
    * we don't really know what the right outcome is
* types of clustering
    * partitional: each object belongs to exactly one cluster
    * hierarchical: set ofnested clusters organized in a tree (like phylogenetic tree)
    * density based: defined based on local density of points
    * soft clustering: each point is assigned to every cluster with a certain probability
* so what makes well partitioned clusters?
    * when partitions ar every overlaped -> high variance (greater spread aroudn the mean)
    * ![image info](./assets/clusters.png)
* cost function: way to evaluate and compare solutions
    * $\sum_{i}^{k} \sum_{x∈C}^{} d(X, \mu_i)^2$
        * where $\sum_{x∈C}^{} d(X, \mu_i)^2$ (right sum) is the variance
    * we want to find some algo that can find solutions to minimize cost
    * so like to minimize cost you have to minimize variance...?
* one way to reduce cost
    * if the "blue" points are closer to the yellow mean than the blue mean, then it's better for cost to reduce the spread of the blue cluster and assign those points to the yello wmean than trying to expand the yellow???
    * so like basically just adjust the partition by giving points from one cluster to another depending on which center those clusters are closer to
* what defines a cluster? the center!
* so how do we actually go about this alog? what algo to use? -> lloyd's algo
    * randomly place (assign) k centers (and thus make k clusters)
    * asign each point in the dataset to the closest cluster
        * don't get it twisted, you place the point in the closest cluster but that doesn;t mean the center you assigned it to is still the actual center of the cluster, it was just the closest
    * computer the new centers as the emans of each clust
    * repeat 2 & 3 until convergence
        * convergence => centers stop changing
* how do we know how many clusters.....
    * we have to figure out how to et algorithm to tell us that
* will this algo always lead to convergence? we'll see
* lloyds algorithm is not alway optimal!!
    * imagine that the true center lies righ tin the middle of two blue clusters, those clusters will neve come together even though they the same, so not optimal

### 9/23/24
* finishing lloyd's algo
    * prof way: pseudo code with what methods you'll need, then implement those functions
    * see workseet 3 for lloyds
* will lloyd's always converge? let's prove by contradiction
    * suppose it doesn't converge, then either
        * the minimum of the cost function is only reached in the limit (i.e. an infinite number of iterations, like you tryign to look at every partition ever and there are infinite??)
            * but impossibel since we are iterating over a finite set of partitions
        * the algo gets stuck in a cycle or a loop
            * not possible since this would only be possible when if you have 2 overlapping points and a randomized assignment of points to clusters, but our algo can spot that these are the same points
* does it always converge to optimal solution though?
    * no, there can be clusters that are close but don't necessarily count as an organic organic cluster as one but maybe as two
* i misssed some stufff
* k-means++: this is a combination of the two methods (decreasing randomization and choosing a point farther from ????????????????????????)
    * start with random center
    * let d(x) be the distance between x and the closest of the centers picked so far
        * choose the next center with probability proportional to $d(x)^2$
            * this allows for less randomization and for greater distance point to be more likely to be picked
* bro im crashing out
* so how do we choose the right k?
    * iterate through different values of k
    * use empirical/domain specific knowledge
    * metric for evaluating a clustering output
* so remember our goal to find a clustering such that
    * similar dtaa points are in the sae cluster
    * dissimilar data points are in different clusters
    * we wanna create clusters that are far from each other, clusters should be distinct from each other
* how could we try to define this metric that evaluates how spread out the clusters are from one another
    * maybe distance between centers
    * maybe minimum distance between points across clusters
    * make the cneter distnce but then subract the sread within one cluster, since maybe need to account for that?
* let b be the distance between centers and a be the dustanc across one center
    * if b - a = 0, this means that the clusters overlao
    * so ideally we want b - a to be large!! this means clusters are spread far apart relative to the compactness of the clusters
* but the value of b - a doesn;t really mean anything so how do we really get a meaning
    * like b - a = 5 doesn't tell us about anything
    * so do $\frac{(b - a)}{max(a, b)}$
        * if $\frac{(b - a)}{b} = 1$ then b - a is basically b, so a is really small, so that means clusters are small and there is good spread
        * if $\frac{(b - a)}{max(a, b)} = 0$, then there may be overlap
* sillhouette scores
    * for each data point i, define:
        * $a_i$: ean distnce from point i to ever other point in its cluster
        * $b_i$ smallest mean distance from point i to every point in another cluster
    * so silhouette score $s_i$ = $\frac{b_i - a_i}{max(a_i, b_i)}$
        * low silhouette score is bad...
        * you can plot silhouette score of plo to get an idea
    * how to tell if silhouette score is better...?
        * avg silhouette score across clusters good
        * clusters are all kind of similar
* point of diminishing return...
    * silhouette score plot bad if pqast the point of diminishig return

### 9/25/24
* dendrogram: tels you at what distance do you stat to merge points together becuse they are so similar
    * kind of like a phylogenetic tree
* can cut the dendrogram (like horiztonal line) to get a number of clusters
    * ![image info](./assets/hierarchical.png)
* 2 types of hierarchical
    * agglomerative
        * start with every point in its own cluser
        * at each step merge the two clsoest clusters
        * stop when every point is in the same cluster
    * divisive
        * start wtih every point in the same cluster
        * at each step split until each point is in its own cluster
    * but what's our metric for merging clusters??
* to actual split our clusters we can use kmeans
* single link distance: the minimum of all pairwise distances between a point from one cluster and a point from another cluster
    * $D_{SL}(C_1, C_2) = min\{d(p_1, p_2) | p_1 ∈ C_1, p_2 ∈ C_2\}$
    * basically look at all the possible distances between a pointin cluster A and a point in cluster B, then take the minimum distance
* so we use single link distance to decide which point to merge next into our cluster (agglomerative)
    * so if point D is closer to any point in the cluster than point C is to any pointi n the cluster, than merge D first
    * single link distance is from D, not C
* complete link distance: basically single link but now with the max pairwise distance
    * $D_{SL}(C_1, C_2) = max\{d(p_1, p_2) | p_1 ∈ C_1, p_2 ∈ C_2\}$
* i think
    * complete link distance allows us to find the distance between clusters in hierarchical clustering
    * single link distance to decide which point to merge
* average link distance: all pairwise distances, then find the average of them all
    * formula...
* centroid distance: find the distance between centroids of clusters
* ward's distance: difference between the spread/variance of pointsin  the merged clusters and the unmerged clusters
    * answers the question of what the consequence of merging points does to the variance
        * merging points can increase variance, so ward's distance lets ys see that
    * see slides for formula
* so actual agglomerative clustering algorithm
    * let each point in the dataset be in its own clustering
    * compute the distance between all pairs of clusters
    * merge the two closest clusters
    * repeat 3 & 4 until all points are in the same cluster
* density based clustering
    * dense if contains a certain amount of data points within some radius of region
* we shoudl dstinguish between poitns at the core of a dense refdion and points at the border of the dense region
    * core point: a point that has at least min_pts points withint its defined neighborhood
    * noise point: neither a core nor border (doesn't have enough points around it)
    * boider point:if it is in the neighborhood of a core point
* DBScan algorithm, given the size of the $\eps$neighborhood and min_pts
    * find the neighborhood of each point 
    * label the point as core if its enighborhood has at least min_pts
    * for each core pint, assign to the same cluster all the core points in its neighborhood
    * label points in its neighborhood that are not core as broder
    * label points as noise if neither core nor border
    * assign bordeer points to nearby clusters
* kind of like DFS, liek basically look at the nighbors of a core point and decide if they are core poitns,a nd if core then same cluster then repeat
    * basically you just look at a core and add all the core points in the neighborhood to the cluster, and like border points in the cluster ar elebaled as border, but are still in cluster
* beneifits of DBScan
    * can identify clusters of difference shapes and sizes
    * resistant to noise
* 