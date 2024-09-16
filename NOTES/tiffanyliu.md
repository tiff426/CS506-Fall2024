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
  
  
