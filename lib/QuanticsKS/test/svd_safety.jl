module SVDSafetyChecks
using Test, QuanticsKS, LinearAlgebra, Random, Serialization, JSON3
const Q = QuanticsKS.Q
const rng = MersenneTwister(7821)
BLAS.set_num_threads(1)
function train(R, rank=3)
    ranks = [1; fill(rank, R-1); 1]
    Q.QTT([randn(rng, ComplexF64, ranks[k], 2, ranks[k+1]) for k in 1:R])
end
records(dir) = [deserialize(joinpath(dir,f)) for f in readdir(dir) if endswith(f,".jls")]

@testset "Checked compression preserves pinned library results" begin
    mktempdir() do dir
        Q.with_svd_context(; diagnostic_dir=dir) do
            for R in (1,5,9), tol in (0.0,1e-6), cap in (1,16)
                x=train(R); saved=deepcopy(x.c)
                expected=deepcopy(Q.tensortrain(x))
                Q.TCI.compress!(expected,:SVD;tolerance=tol,maxbonddim=cap)
                actual=Q.rounded(x;tol,cap)
                @test actual.c == expected.sitetensors
                @test x.c == saved
            end
            x,y=train(6),train(6)
            expected=Q.TCI.add(Q.tensortrain(x),Q.tensortrain(y);
                factorlhs=2im,factorrhs=-3,tolerance=1e-6,maxbonddim=4)
            @test Q.add(x,y;a=2im,b=-3,tol=1e-6,cap=4).c == expected.sitetensors
            frame=Q.canonical_frame(x)
            right=deepcopy(Q.tensortrain(x)); Q.TCI.compress!(right,:SVD;tolerance=0.0)
            left=reverse(Q.tensortrain(x)); Q.TCI.compress!(left,:SVD;tolerance=0.0)
            @test frame.right == right.sitetensors
            @test frame.left == reverse(left).sitetensors
            @test isempty(readdir(dir))
        end
    end
end

@testset "Finite SVD failures recover from the preserved input" begin
    A=randn(rng,ComplexF64,7,4); saved=copy(A)
    for kind in (:lapack,:argument4,:nonfinite_U,:nonfinite_S,:nonfinite_Vt)
        mktempdir() do dir
            calls=Symbol[]
            function backend(B;full,alg)
                push!(calls,alg isa LinearAlgebra.QRIteration ? :qr : :divide)
                if alg isa LinearAlgebra.DivideAndConquer
                    kind==:lapack && throw(LinearAlgebra.LAPACKException(1))
                    kind==:argument4 && throw(ArgumentError("invalid argument #4 to LAPACK call"))
                    f=svd(B;full,alg)
                    kind==:nonfinite_U && (f.U[1]=NaN)
                    kind==:nonfinite_S && (f.S[1]=NaN)
                    kind==:nonfinite_Vt && (f.Vt[1]=NaN)
                    return f
                end
                svd(B;full,alg)
            end
            result=Q.with_svd_context(;diagnostic_dir=dir,scf=8,stage="L6",orbital=46) do
                Q.checked_svd(A;factorize=backend)
            end
            @test calls == [:divide,:qr]
            @test result.U*Diagonal(result.S)*result.Vt ≈ A
            @test A == saved
            event=only(records(dir))
            @test event.matrix == A
            @test event.context.scf==8 && event.context.orbital==46
            summary=JSON3.read(read(only(filter(f->endswith(f,".json"),readdir(dir;join=true))),String))
            @test summary.status == "recovered_with_qr"
        end
    end
end

@testset "Nonfinite inputs are saved and never sent to QR" begin
    mktempdir() do dir
        calls=Ref(0)
        backend(A;kwargs...)=(calls[]+=1;svd(A;kwargs...))
        A=fill(ComplexF64(NaN),3,3)
        Q.with_svd_context(;diagnostic_dir=dir) do
            @test_throws ErrorException Q.checked_svd(A;factorize=backend)
        end
        @test calls[]==0
        @test isequal(only(records(dir)).matrix,A)
    end
    mktempdir() do dir
        x=train(4); x.c[3][1]=Inf
        Q.with_svd_context(;diagnostic_dir=dir,orbital=46,role="history_direction") do
            @test_throws ErrorException Q.rounded(x)
        end
        e=only(records(dir))
        @test e.context.site==3 && e.context.operation=="rounded"
        @test isequal(e.context.original_cores,x.c)
        @test e.context.role=="history_direction"
    end
    mktempdir() do dir
        x=Q.QTT([fill(ComplexF64(1e200),1,2,1) for _ in 1:2])
        Q.with_svd_context(;diagnostic_dir=dir) do
            @test_throws ErrorException Q.rounded(x)
        end
        e=only(records(dir))
        @test e.reason=="nonfinite_forward_contraction"
        @test all(isfinite,e.details.right) && all(isfinite,e.details.neighbor)
        @test !all(isfinite,e.matrix)
    end
end

@testset "Exception boundaries and diagnostic context lifetime" begin
    mktempdir() do dir
        Q.with_svd_context(;diagnostic_dir=dir) do
            A=ones(ComplexF64,2,2)
            calls=Ref(0)
            unrelated(A;kwargs...)=(calls[]+=1;throw(ArgumentError("unrelated contract violation")))
            @test_throws ArgumentError Q.checked_svd(A;factorize=unrelated)
            @test calls[]==1 && isempty(readdir(dir))
            failed_qr(A;kwargs...)=throw(LinearAlgebra.LAPACKException(1))
            @test_throws LinearAlgebra.LAPACKException Q.checked_svd(A;factorize=failed_qr)
            summary=JSON3.read(read(only(filter(f->endswith(f,".json"),readdir(dir;join=true))),String))
            @test summary.status=="qr_failed"
        end
        @test !haskey(task_local_storage(),Q.SVD_CONTEXT_KEY)
    end
    mktempdir() do dir
        invalid=Q.QTT([ones(ComplexF64,1,2,2),ones(ComplexF64,3,2,1)])
        Q.with_svd_context(;diagnostic_dir=dir) do
            @test_throws DimensionMismatch Q.rounded(invalid)
        end
        @test only(records(dir)).reason=="invalid_tensor_shape"
    end
    mktempdir() do dir
        A=ones(ComplexF64,3,2)
        function bad_qr(B;full,alg)
            alg isa LinearAlgebra.DivideAndConquer && throw(LinearAlgebra.LAPACKException(1))
            f=svd(B;full,alg);f.S[1]=NaN;f
        end
        Q.with_svd_context(;diagnostic_dir=dir) do
            @test_throws ErrorException Q.checked_svd(A;factorize=bad_qr)
        end
        @test length(records(dir))==2
        @test any(e->e.reason=="nonfinite_qr_output",records(dir))
    end
    mktempdir() do dir
        A=ones(ComplexF64,3,2)
        backend(B;full,alg)=alg isa LinearAlgebra.DivideAndConquer ?
            throw(LinearAlgebra.LAPACKException(1)) : svd(B;full,alg)
        Random.seed!(8123); expected=rand()
        Random.seed!(8123)
        Q.with_svd_context(;diagnostic_dir=dir) do
            Q.checked_svd(A;factorize=backend)
        end
        @test rand()==expected
    end
end
end
