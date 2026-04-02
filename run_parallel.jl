using Distributed 
addprocs(4) ##Number of threads used in operations 
println("Available workers: ", workers()) 
###  Entorno asociado ocn los procesos que se corren en paralelo
println("number threads used in operations is : " , Threads.nthreads()  )
@everywhere begin 
    include("./main.jl")
    Ls = [10,20,30,40]
    function results(i1)
        println("Running task for (i1) = ($i1) on worker: ", myid())
        main(;L=Ls[i1])
    end
    i_index = [(i1) for i1 in 1:length(Ls)]
    println("The program started")
    pmap(i -> results(i[1]), i_index )

end