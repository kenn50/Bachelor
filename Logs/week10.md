I only have this saturday to work on the project. I will do as much as i can but i will have more time in a few weeks.

# Making it more simple
The theme of this day is to load a more simple dataset that is easy to debug. I will start over and do one step at a time and make sure it works.


# Fixing the whole problem for toy datasets
I did not attempt SMI yet, but experimented with the toydataset of generating either S-shaped manifolds in 3D or the morbius circle. I found that using the normalizing flow indeed solved the whole problem very effectively. While the other VAE's had massive wholes in their z-distrubution making generation almost impossible, the normalizing flow helped.

# Created single site rbf. 
Taking just one site (such as mean) it calculates the rbf kernel for just this site
