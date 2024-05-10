module RWC
    using ProgressMeter, LinearAlgebra, GraphRecipes,
          SparseArrays, Random, PlotlyJS, Graphs, GraphPlot, Plots, Compose

 mutable struct graph_lattice
    n::Vector{Int64} #number of vertices (x,y direction)
    nV::Int64 #total number of vertices 
    V::Vector{Int64} #all vertices
    B::Vector{Int64} #boundary nodes 
    nB::Int64 #number of boundary nodes 
    I::Vector{Int64} #interior nodes 
    Ω::Vector{Int64} #cloaking region nodes
    ∂Ω::Vector{Int64} #cloaking region boundary nodes
    A::Matrix{Float64} #incidence matrix 
    Lap::Matrix{Float64} #Laplacian matrix 
    R::Matrix{Int} #restriction matrix 
    G::Matrix{Float64} #Green function matrix 
    P::Matrix{Float64} #Transition probability matrix 
    Px::Matrix{Float64} #Calculates the modified probability matrix using DTN 
    Ar::Matrix{Float64} #Modified adjacent matrix using DTN 
    locx::Vector{Float64} #xloc for layout 
    locy::Vector{Float64} #yloc for layout  
 end#struct

 mutable struct graph_setup
   nV::Int64 #total number of vertices 
   V::Vector{Int64} #all vertices
   B::Vector{Int64} #boundary nodes 
   nB::Int64 #number of boundary nodes 
   I::Vector{Int64} #interior nodes 
   Ω::Vector{Int64} #cloaking region nodes
   ∂Ω::Vector{Int64} #cloaking region boundary nodes
   A::Matrix{Float64} #incidence matrix 
   Lap::Matrix{Float64} #Laplacian matrix 
   R::Matrix{Int} #restriction matrix 
   G::Matrix{Float64} #Green function matrix 
   P::Matrix{Float64} #Transition probability matrix 
   Px::Matrix{Float64} #Calculates the modified probability matrix using DTN 
   Ar::Matrix{Float64} #Modified adjacent matrix using DTN 
   locx::Vector{Float64} #xloc for layout 
   locy::Vector{Float64} #yloc for layout 
end#struct


 """
 Generates lattice graph with random (positive weights) 
 """
 function gen_lattice(;n = [11,11],
                      m1 = (4,4),
                      m2 = (8,8))

    #Find all vertex sets 

    nV = prod(n)
    isboundary = [ i1 ∈ [1,n[1]] || i2 ∈ [1,n[2]] for i1=1:n[1], i2=1:n[2] ]
    is∂Ω =[ (i1 ∈ [m1[1],m2[1]] && i2 ∈ m1[2]+1:m2[2]-1 )|| (i2 ∈ [m1[2],m2[2]] && i1 ∈ m1[1]+1:m2[1]-1) for i1=1:n[1], i2=1:n[2] ]
    isΩ = [i1 ∈ m1[1]+1:m2[1]-1 && i2 ∈ m1[2]+1:m2[2]-1 for i1=1:n[1], i2=1:n[2]]
    V = 1:nV
    B = findall(isboundary[:]) 
    I = findall(.~isboundary[:]) 
    ∂Ω = findall(is∂Ω[:])
    Ω = findall(isΩ[:])

    #Generate incidence matrix (could be improved but fine for small examples)
    A = zeros(nV,nV)
    [A[i,j] = 1 for i = 1:nV for j = 1:nV if abs(i-j)==1] #horizontal connections
    [A[i,j] =0 for i = 1:nV for j = 1:nV if (mod(j,n[1])==0 && mod(i,n[2])==1 )]
    [A[i,j] =0 for i = 1:nV for j = 1:nV if (mod(i,n[1])==0 && i<j)] #delete right edges
    [A[i,j] =0 for i = 1:nV for j = 1:nV if (mod(i,n[1])==1 && j<i)] #delete left edges 
    [A[i,j] = 1 for i = 1:nV for j = 1:nV if (mod(i,n[1])-mod(j,n[1]) ==0 && abs(i-j)<n[1]+1)] #vertical connections
    [A[i,j] =0 for i = 1:nV for j = 1:nV if i == j] #delete self connections 
  
    A = (A'+A)/2  

    #Laplacian matrix 
    Lap = A - diagm(A*ones(size(A,1)))

    #Restriction matrix 
    R = spzeros(nV, size(∂Ω,1))
    for i ∈ 1:size(∂Ω,1)
        R[∂Ω[i],i] = 1
    end 

    #Green function matrix (w/ zero Dirichlet)
    G = zeros(nV,nV)
    Vm∂F = setdiff(V,B)
    G[Vm∂F,Vm∂F] = inv(Matrix(-Lap[Vm∂F,Vm∂F]))

    #Transition prob matrix 
    DL = -diagm(1 ./ diag(Lap))
    P = DL * Lap  + LinearAlgebra.I 

    #Modified using DTN 
    
    DTN = Lap[∂Ω,∂Ω] - Lap[∂Ω,Ω] * (Lap[Ω ,Ω]\Lap[Ω,∂Ω])
    
    Lr = deepcopy(Lap)
    Lr[∂Ω,∂Ω] = DTN
    Lr[setdiff(V,Ω),Ω] .=0
    Lr[Ω,setdiff(V,Ω)] .=0
    Lr[Ω,Ω] .=0

    # transition proba matrix
    DL = -diagm(1 ./ diag(Lr))
    #DL = -diagm( [x!=0 ? 1/x : 0 for x ∈ diag(Lr)] )

    Px = DL * Lr  + LinearAlgebra.I 

    Ar = deepcopy(Lr); Ar = Ar - diagm(diag(Ar));

    g = SimpleGraph(nV)
    [add_edge!(g, i,j ) for i = 1:nV for j =1:nV if A[i,j] != 0 ]

    locx, locy = spring_layout(g)

    return graph_lattice(n,nV,V,B,size(B,1),I,Ω,∂Ω,A,Lap,R,G,P,Px,Ar,locx,locy,)
 end#function 

 """
 Adds a defective node to the middle of the lattice
 """
 function add_defect(gd,defect_node)
   gd2 = deepcopy(gd)
   n = gd2.n
   nV = gd2.nV 
   V = gd2.V 
   B = gd2.B ∪ defect_node 
   I = setdiff(gd2.I, defect_node)
   ∂Ω = gd2.∂Ω
   Ω = gd2.Ω
   A = gd2.A 
   Lap = A - diagm(A*ones(size(A,1)))
   R = gd2.R

   #Green function matrix (w/ zero Dirichlet)
   G = zeros(nV,nV)
   Vm∂F = setdiff(V,B)
   G[Vm∂F,Vm∂F] = inv(Matrix(-Lap[Vm∂F,Vm∂F]))

   #Transition prob matrix 
   DL = -diagm(1 ./ diag(Lap))
   P = DL * Lap  + LinearAlgebra.I 

   #Modified using DTN 
   
   DTN = Lap[∂Ω,∂Ω] - Lap[∂Ω,Ω] * (Lap[Ω ,Ω]\Lap[Ω,∂Ω])
   
   Lr = deepcopy(Lap)
   Lr[∂Ω,∂Ω] = DTN
   Lr[setdiff(V,Ω),Ω] .=0
   Lr[Ω,setdiff(V,Ω)] .=0
   Lr[Ω,Ω] .=0

   # transition proba matrix
   DL = -diagm(1 ./ diag(Lr))
   #DL = -diagm( [x!=0 ? 1/x : 0 for x ∈ diag(Lr)] )

   Px = DL * Lr  + LinearAlgebra.I 

   Ar = deepcopy(Lr); Ar = Ar - diagm(diag(Ar));

   g = SimpleGraph(nV)
   [add_edge!(g, i,j ) for i = 1:nV for j =1:nV if A[i,j] != 0 ]

   locx, locy = spring_layout(g)
   return graph_lattice(n,nV,V,B,size(B,1),I,Ω,∂Ω,A,Lap,R,G,P,Px,Ar,locx,locy)
 end

 """
 Generates the paper example for a graph 
   In this case nI denotes the number of interior nodes that do not contain Ω and ∂Ω
 """
 function gen_paper_example(;nB=6,nI = 6, nΩ=2, n∂Ω=3, p = [0.85, 0.8])
   #Define nodes 
   B = 1:nB; 
   I = nI .+ (1:nI)
   ∂Ω = nB+nI .+ (1:n∂Ω)
   Ω = nB+nI+n∂Ω .+ (1:nΩ)
   nV = nB + nI + nΩ+ n∂Ω
   V = 1:nV
   A = zeros(nV,nV)

   Lap = zeros(nV,nV)
   
   function gen_connections(A, sets,x)
      for i ∈ sets[1], j ∈ sets[2]
         A[i,j] =  (1+rand())*(rand()>x)  # here i>j
      end
   end 

   for X ∈ [ [B,B],[I,I],[∂Ω,∂Ω]]
      gen_connections(A, X, p[1])
   end 

   for X ∈ [[I,B],[I,∂Ω]]
      gen_connections(A, X, p[2])
   end 

   gen_connections(A,[∂Ω, Ω[2]],0)

   gen_connections(A,[Ω[1],Ω[2]],0)
   for i =1:nV
      A[i,i] =0
   end
   I = setdiff(V,B)
   A = A+A'

   Lap = A - diagm(A*ones(size(A,1))) # graph Laplacian

   #Restriction matrix 
   R = zeros(nV, size(∂Ω,1))
   for i ∈ 1:size(∂Ω,1)
      R[∂Ω[i],i] = 1
   end
   
   #Green function matrix (w/ zero Dirichlet)
   G = zeros(nV,nV)
   Vm∂F = setdiff(V,B)
   G[Vm∂F,Vm∂F] = inv(Matrix(-Lap[Vm∂F,Vm∂F]))

   #Transition prob matrix 
   DL = -diagm(1 ./ diag(Lap))
   P = DL * Lap  + LinearAlgebra.I 
   
       #Transition prob matrix 
    DL = -diagm(1 ./ diag(Lap))
    P = DL * Lap  + LinearAlgebra.I 

    #Modified using DTN 
    DTN = Lap[∂Ω,∂Ω] - Lap[∂Ω,Ω] * (Lap[Ω ,Ω]\Lap[Ω,∂Ω])
    Lr = deepcopy(Lap)
    Lr[∂Ω,∂Ω] = DTN
    Lr[setdiff(V,Ω),Ω] .=0
    Lr[Ω,setdiff(V,Ω)] .=0
    Lr[Ω,Ω] .=0

    # transition proba matrix
    DL = -diagm(1 ./ diag(Lr))
    Px = DL * Lr  + LinearAlgebra.I
    
    Ar = deepcopy(Lr); Ar = Ar - diagm(diag(Ar));
    locx = []; locy = [];

    #Locations for plotting
    [push!(locx,x[1]) for x ∈ circ_points(nB, [0,0])]
    [push!(locx,x[1]) for x ∈circ_points(nI, [3,0])]
    [push!(locx,x[1]) for x ∈circ_points(n∂Ω, [6,0])]
    [push!(locx,x[1]) for x ∈circ_points(nΩ, [9,0])]
    [push!(locy,x[2]) for x ∈ circ_points(nB, [0,0])]
    [push!(locy,x[2]) for x ∈circ_points(nI, [3,0])]
    [push!(locy,x[2]) for x ∈circ_points(n∂Ω, [6,0])]
    [push!(locy,x[2]) for x ∈circ_points(nΩ, [9,0])]
   
   return graph_setup(nV, V, B, nB, I, Ω, ∂Ω, A,Lap,R,G,P, Px, Ar,locx,locy)
 end#function

 
 """
 Laplacian on restricted set of nodes 
 """
 function lap(E,gd) 
    LEE = gd.A[E,E] - spdiagm(gd.A[E,E]*ones(length(E)))
    L = spzeros(gd.nV,gd.nV)
    L[E,E] = LEE
    return L
 end
 SLP(E1,E2,gd) = gd.G*gd.R*gd.R'*(lap(E1,gd)-lap(E2,gd)/2) 
 DLP(E1,E2,gd) = gd.G*(lap(E1,gd)-lap(E2,gd)/2)*gd.R*gd.R'
 SLP_noG(E1,E2,gd) = gd.R*gd.R'*(lap(E1,gd)-lap(E2,gd)/2) 
 DLP_noG(E1,E2,gd) = (lap(E1,gd)-lap(E2,gd)/2)*gd.R*gd.R'


 """
 RW solution of the Dirichlet problem 
 """
 function RW_dirichlet(gd,u,Nrel)
   u_mc = zeros(gd.nV) 
   cx = gd.A*ones(gd.nV) #total conductance at each node 
   for n ∈ 1:Nrel 
      for i ∈ gd.B
        charge = u[i]*cx[i] #keeps track of where the particle started 
        u_mc[i] += charge
        z = rand()
        j = findfirst(x->x>=z,cumsum(gd.P[i,:]))  
        while !(j ∈ gd.B)
            try
             u_mc[j] += charge
            catch
               return gd
            end
            z = rand()
            j = findfirst(x->x>=z,cumsum(gd.P[j,:]))     
        end
      end
   end
   return u_mc./Nrel./cx   
 end

 """
 Stochastic realization of SLP and DLP from RW perspective 
 """
 function stochastic_rep(gd,Nrel,w)
 cx = gd.A*ones(gd.nV) #total conductance at each node 
 urec = zeros(gd.nV)
 wloc = findall(w.!=0)[:]
 for n ∈ 1:Nrel
    for m ∈ 1:size(wloc,1)
        j = wloc[m] # start point of random walk
        while !(j ∈ gd.B) #  while we are not at end point of random walk
            urec[j]+= w[wloc[m]]/cx[j]
            z = rand()
            jnew = findfirst(x->x>=z,cumsum(gd.P[j,:]))   
            j = jnew
        end
    end
 end

 return urec = urec / Nrel
 end#function 

 """
 Stochastic realization of the DTN from the RW perspective 
 """
 function perfectcloak(gd,Nrel,u)
   cx = gd.Ar*ones(gd.nV)
   u_mc = zeros(gd.nV)
   for n ∈ 1:Nrel 
      for i ∈ gd.B
          charge = u[i]*cx[i] #keeps track of where the particle started 
          u_mc[i] += charge
          z = rand()
          j = findfirst(x->x>=z,cumsum(gd.Px[i,:]))
          while !(j∈gd.B)
              u_mc[j] += charge
              z = rand()
              j = findfirst(x->x>=z,cumsum(gd.P[j,:])) 
          end
          #if current_iter == max_iter  println("Max iter hit") end
      end
  end
  u_mc = u_mc./Nrel./cx
  u_mc[gd.Ω] .=0
  return u_mc
 end

 """
 Determine nodal control 
 """
 function calc_sets(gd)
   ms = gd.∂Ω ∪ gd.Ω; bs= gd.∂Ω; ps = setdiff(gd.V,gd.Ω);
   wm = (-SLP_noG(ms,bs,gd)+DLP_noG(ms,bs,gd))*ones(gd.nV)
   wp = (SLP_noG(ps,bs,gd)-DLP_noG(ps,bs,gd))*ones(gd.nV)
   control_plus =  zeros(gd.nV); control_minus = zeros(gd.nV)
   control_plus[wp.!=0] .= 1; control_minus[wm.!=0] .= 1
   return control_plus, control_minus
 end

 """
 Points on a circle at center 
 """
 function circ_points(n, ctr; radius=1/2)
   circ_points = []
   Θ = 2*π*(1 .- 1:(n-1))./n
   [push!(circ_points, [radius*cos(θ) + ctr[1], radius*sin(θ) + ctr[2]]) for θ ∈ Θ ]
   return circ_points
 end
 
 """
 Plots graph function as color 
 """
 function plot_lat_function(gd,u; save = false, filename="fig", lab = "no")
   pos_x = []
   for i  ∈ 1:gd.n[2]
      clist = 1:gd.n[2]
      if mod(i,2) == 0
         clist = gd.n[2]:-1:1
      end
      for x ∈ clist
         push!(pos_x,x)
      end
   end

   pos_y = repeat(1:gd.n[1],inner = gd.n[2])
 
   edge_x = pos_x
   edge_y = pos_y
   
   
   color_map = u
   # Create edges
   
   edges_trace1 = PlotlyJS.scatter(
       mode="lines",
       x=edge_x,
       y=edge_y,
       line=attr(
           width=1,
           color="#888"
       ),
   )

   edge_trace2 = PlotlyJS.scatter(
      mode="lines",
      x=edge_y,
      y=edge_x,
      line=attr(
          width=1,
          color="#888"
      ),
  )
   
   
   # Create nodes
   colorbar = true
   w=500; h=500;
   if save == "yes"
      colorbar = false; w=500; h = 500;
   end
   if lab == "yes"
      colorbar = true
   end
   nodes_trace = PlotlyJS.scatter(
       x=pos_x,
       y=pos_y,
       mode="markers",
       text = [string("Net particle charge: ", connection) for connection in color_map],
       marker=attr(
           showscale=colorbar,
           colorscale=colors.YlOrRd,
           #reversescale = true,
           size=16,
           color=color_map,
           colorbar=attr(
               thickness=15
         )
       )
   )
   
   
   # Create Plot
   p = PlotlyJS.plot(
       [edges_trace1,edge_trace2, nodes_trace],
       Layout(
           hovermode="closest",
           width = w,
           height = h,
           showlegend=false,
           showarrow=false,
           xaxis=attr(showgrid=false, zeroline=false, showticklabels=false),
           yaxis=attr(showgrid=false, zeroline=false, showticklabels=false, scaleanchor="x"),
           margin = attr(l=0, #left margin
           r=0, #right margin
           b=0, #bottom margin
           t=0) #top margin
       )
   )
   if save == "yes"
      PlotlyJS.savefig(p, filename)
   end
   return p
 end 

 """
 Makes ∂Ω a complete graph and removes edges connection it to Ω
 """
 function modify_DTN(gd)
   A_new = deepcopy(gd.A) 
   [A_new[i,j] = 0 for i ∈ gd.Ω for j ∈ gd.∂Ω]
   [A_new[i,j] = 0 for i ∈ gd.∂Ω for j ∈ gd.Ω]
   [A_new[i,j] = 1 for i ∈ gd.∂Ω for j ∈ gd.∂Ω]
   [A_new[i,i] =0 for i ∈gd.∂Ω ]
   return A_new
 end#function 

 """
 Plots graph functions in color
 """
 function plot_graph_func(gd,u; save = "false", filename = "fig",labels = "yes", DTN_graph = "no")
   g = SimpleGraph(gd.nV)
   if DTN_graph == "yes"
      A_new = modify_DTN(gd)
      [add_edge!(g, i,j ) for i = 1:gd.nV for j =1:gd.nV if A_new[i,j] != 0 ]
   else
      [add_edge!(g, i,j ) for i = 1:gd.nV for j =1:gd.nV if gd.A[i,j] != 0 ]
   end
   nodefillc = []
   [push!(nodefillc, cgrad(:heat)[x]) for x ∈  u./maximum(u)]
   size = (1000,250)
   an_size = 14
   if save == "yes"
      size = (4000,1000)
      an_size = 50
   end 
   p= graphplot(g,x = gd.locx, y= gd.locy, nodecolor = nodefillc,nodesize = 1.0, size = size, annotationfontsize=an_size)
   if labels == "yes"
      annotate!(0.0,-1,"B")
      annotate!(3.0,-1,"I")
      annotate!(6.0,-1,"∂Ω")
      annotate!(9.0,-1,"Ω", fontsize = 100)
   end

   if save == "yes"
      Plots.savefig(p, filename)
   end
   return p
 end
  
end#module 