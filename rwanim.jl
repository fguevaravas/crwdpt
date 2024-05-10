# Random walk animations
module rwanim
 using LinearAlgebra, SparseArrays, PyPlot, Printf, FFMPEG, ProgressMeter, Random
 ⊗ = kron
 zerosum(A) = A - spdiagm(A*ones(size(A,1))) 
 
 function generate_lattice(n1,n2,α=0.3)

    V = reshape(1:n1*n2,n1,n2) # vertices
    nV = n1*n2
    # vertex coordinats (uniformly spacd on [0,1]^2)
    x1 = [i1-1 for i1=1:n1, i2=1:n2]/(n1-1)
    x2 = [i2-1 for i1=1:n1, i2=1:n2]/(n2-2)
    
    # adjacency matrix 1d
    A1d(n) = spdiagm(-1=>ones(n-1),1=>ones(n-1))

    # adjacency matrix 2d
    A = I(n2) ⊗  A1d(n1) + A1d(n2) ⊗ I(n1)

    # boundary
    B = [ i1 ∈ [1,n1] || i2 ∈ [1,n2] for i1=1:n1, i2=1:n2 ]
    B = findall(vec(B))

    # cloaked region
    rΩ = 0.25
    Ω = [   norm([x1[i1,i2]-0.5,x2[i1,i2]-0.5],2)<=rΩ for i1=1:n1,i2=1:n2]
    Ω = findall(vec(Ω))

    # cloaked region boundary
    ∂Ω = [ rΩ<norm([x1[i1,i2]-0.5,x2[i1,i2]-0.5],2)<=(rΩ+1/max(n1,n2)) for i1=1:n1,i2=1:n2]
    ∂Ω = findall(vec(∂Ω))

    𝐈  = setdiff(1:nV,B∪Ω∪∂Ω)
    Vᵒ = setdiff(1:nV,B)

    P = diagm(1 ./ (A*ones(n1*n2))) * A # transition probability matrix
    cP = cumsum(P,dims=2) # cumulative probability densities for each node
    c = A*ones(n1*n2) # total charges at node
    L = -zerosum(A) # Laplacian (positive definite one)

    # exterior Laplacian (zero inside Ω)
    Le = copy(L);
    Le[Ω,:] .= 0; Le[:,Ω] .= 0
    Le[∂Ω,∂Ω] .*= 1-α
    Le = zerosum(Le);

    # interior Laplacian (zero outside Ω, i.e. B∪𝐈)
    Li = copy(L);
    Li[B∪𝐈,:] .= 0; Li[:,B∪𝐈] .=0 ;
    Li[∂Ω,∂Ω] .*= α
    Li = zerosum(Li); 

    R = I(nV)[∂Ω,:] # restriction to ∂Ω
    γ0 = R          # Dirichlet trace operator
    γ1e = - R*Le    # exterior Neumann trace operator
    γ1i =   R*Li    # interior Neumann trace operator
    

    return (x1=vec(x1),x2=vec(x2),
            B=B,𝐈=𝐈,Ω=Ω,∂Ω=∂Ω,Vᵒ=Vᵒ,V=1:n1*n2,
            γ0=γ0, γ1e=γ1e, γ1i=γ1i,
            A=A,L=L,P=P,cP=cP,c=c,n1=n1,n2=n2)
 end

 function plot_graph(G,f,cmap=ColorMap("bwr"))
    scatter(G.x1,G.x2,c=f,s=50,cmap=cmap)
 end

 function test_generate_lattice(n1=5,n2=7)
    G = generate_lattice(n1,n2)
    f = randn(n1*n2)
    plot_graph(G.x1,G.x2,f,G.A)
 end

 # a random walker is a named tuple with entries:
 #  i=position as a linear index in the lattice
 #  q=charge
 # this function advances one random walker in the graph G
 function advance(G,rw)
   z = rand()
   return (i=findfirst(x->x>=z,G.cP[rw.i,:]),
           q=rw.q)
 end

 # identifies all the walkers that reach the boundary
 # and deletes them
 delete_rw(rws,B) = [ rw for rw in rws if rw.i ∉ B] 

 # for debugging purposes
 function print_walkers(rws,cidx)
  for rw ∈ rws
    c = cidx[rw.i]
    print(" ($(c[1]),$(c[2]))")
  end
  println()
 end

 # neighborhood of set S in graph
 function neighborhood(S,G)
   u = [ i ∈ S for i = 1:length(G.V) ]
   NS = S ∪ findall(abs.(G.A*u) .> 1e-10)
 end

 # n1,n2 = dimensions of lattice
 # nsteps = total number of frames
 # T = how far a part batches of walkers are released
 # k = how many batches of walkers are released at a time
 # transitions: whether to plot the transitions
 # note: running this function generates nsteps frames (png files) in directory `frames`
 # which is created automatically. This can take a lot of space, but is greatly compressed
 # defaults: (n1=20,n2=20,nsteps=600,T=10,k=100)
 function do_animation(n1=20,n2=20,nsteps=1200,T=100,k=50)
   # create directory for frames (ignoring errors)
   try mkdir("frames") catch end

   # initialize random seed for reproducibility
   Random.seed!(4)

   # generate graph and extract trace operators
   G = generate_lattice(n1,n2)
   B = G.B; Vᵒ = G.Vᵒ; ∂Ω = G.∂Ω; Ω = G.Ω
   γ0 = G.γ0; γ1e = G.γ1e

   # set boundary condition to x1 + x2
   uB = G.x1[B] + G.x2[B]

   # calculate average solution
   u = zeros(n1*n2)
   u[B] = uB 
   u[Vᵒ] = -G.L[Vᵒ,Vᵒ]\(G.L[Vᵒ,B]*u[B])

   # calculate active set (A+)
   A = setdiff(neighborhood(G.∂Ω,G) , Ω) 
   density = (γ0'*γ1e - γ1e'*γ0)*u
   dA = -density[A] # the minus sign is here because we want to cloak
   println("|A| = $(length(A)), |B| = $(length(B))")

   # batches of walkers starting at A and B
   B_newbatch = [ (i=v,q=G.c[v]*uB[i]) for (i,v) ∈ enumerate(B) ]
   A_newbatch = [ (i=v,q=dA[i]) for (i,v) ∈ enumerate(A) ]
   
   # list of current random walkers (we keep track of those starting from B and A separately)
   rwB = []; rwA = [] 

   # to keep track of charges 
   qB = zeros(n1*n2); qA = zeros(n1*n2)
   
   fig = figure(figsize=(15,5),layout="tight")
   @showprogress for i=1:nsteps
      # remove terminated walkers
      rwB = delete_rw(rwB,B)
      rwA = delete_rw(rwA,B)

      # time to add new walkers
      if (mod(i-1,T)==0)
         append!(rwB,repeat(B_newbatch,k))
         append!(rwA,repeat(A_newbatch,k))
      end 
      
      # update charges
      for rw ∈ rwB
      qB[rw.i] += rw.q
      end
      for rw ∈ rwA
      qA[rw.i] += rw.q
      end

      # advance walkers
      rwB = [advance(G,rw) for rw ∈ rwB]
      rwA = [advance(G,rw) for rw ∈ rwA]

      # scale used for averaging
      scale = 1/( (i-1) / T + 1)/k 
      # color scale
      clims = 2*[-1,1]

      # plot only walkers starting from B
      clf()
      subplot(1,3,1)
      plot_graph(G,scale*qB./G.c); axis("equal"); clim(clims...)
      axis("off"); #colorbar()
      title("uncloaked")

      # plot only walkers starting from A
      subplot(1,3,2)
      plot_graph(G,scale*qA./G.c); axis("equal"); clim(clims...)
      axis("off"); #colorbar()
      title("cloaking sources")

      # plot both walkers starting from A and B
      subplot(1,3,3)
      plot_graph(G,scale*(qA+qB)./G.c); axis("equal"); clim(clims...)
      axis("off"); #colorbar()
      title("cloaked")

      suptitle("DeGiovanni and Guevara Vasquez - Cloaking for random walks using a discrete potential theory")

      # save frame
      savefig(@sprintf("frames/frame%07d.png",i-1));
   end
   close(fig)
 end

 function do_video()
   # generate video
   ffmpeg_exe(`-framerate 60 -i frames/frame%07d.png 
               -filter_complex "format=yuv420p"
               -y -vcodec libx264 rwvideo.mp4`)
 end

 function do_animation_and_video()
   do_animation()
   do_video()
 end
end
