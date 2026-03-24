## Still TODO:

- Implement batch processing for any supplied forcing term in CuPy `ndarray`
- Syntax validation for the output of true initial condition function, supplied by user (future compatibility) or 
unenviable dev. 
- Determine the amount of terms in the initial condition (e.g. current initial condition 3
but that includes that for $j=1$ we have $\hat{f}_j=0$) and if the number of measurements $n$ is not specified in the 
CLI not specified, then go until $k=\lceil\frac{n}{2}\rceil$.
- Sum the $\hat{f}_j$ terms in the present true initial condition and compare against restriction defined by the Sobolev
space restriction $\lbrace f(x)\in\mathscr{F}_r \; | \; \sum^{\infty}_{j=1}{j^{2r} |\hat{f}_j|^2\leq 1} \rbrace$ to 
make sure it is smooth enough where $r\geq 1$.
- Ability to input $r$ into the CLI, used in implementing th main theorem's convergence bound seen in the error
described by $||\left(f(x)-\bar{f}_n(x)\right)||_{L^2(0,\pi)}\leq C(x_0, r)n^{-2r}$.
- Add functionality for input of patterns with specified k-mode limits, e.g. $\frac{1}{j^3}$ or $\frac{1}{2j^2}$ for
some $k$-terms if they are specified. 