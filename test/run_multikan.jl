@testset "Test whether the dimensions of KAN input and output layers are correct" begin
    @testset "Test whether the dimensions of KAN input and output layers are correct without introducing multiplication operators" begin
        width = [3, 5, 3, 2]
        mult_arity = [Dict(), Dict()]
        layer_cfg = [Dict(:grid_len => 2), Dict(:grid_len => 2), Dict(:grid_len => 2)]
        layer_type = KDense
        model = HydroKAN.KAN(width, mult_arity, layer_cfg, layer_type)
        @test model.in_dims == 3
        @test model.out_dims == 2
        @test model.layers.layer_1.in_dims == 3
        @test model.layers.layer_1.out_dims == 5
        @test model.layers.layer_2.in_dims == 5
        @test model.layers.layer_2.out_dims == 3
        @test model.layers.layer_3.in_dims == 3
        @test model.layers.layer_3.out_dims == 2
    end

    @testset "Test whether the dimensions of KAN input and output layers are correct when multiplication operators are introduced" begin
        width = [3, 5, 3, 2]
        mult_arity = [Dict(1 => 2, 2 => 3), Dict(1 => 2, 2 => 2)]
        layer_cfg = [Dict(:grid_len => 2), Dict(:grid_len => 2), Dict(:grid_len => 2)]
        layer_type = KDense
        model = HydroKAN.KAN(width, mult_arity, layer_cfg, layer_type)
        @test model.in_dims == 3
        @test model.out_dims == 2
        @test model.layers.layer_1.in_dims == 3
        @test model.layers.layer_1.out_dims == 8
        @test model.layers.layer_2.in_dims == 5
        @test model.layers.layer_2.out_dims == 5
        @test model.layers.layer_3.in_dims == 3
        @test model.layers.layer_3.out_dims == 2
    end

    @testset "Test whether the dimensions of KAN input and output layers are correct when there are mixed multiplication and addition operators" begin
        width = [3, 5, 3, 2]
        mult_arity = [Dict(1 => 2, 2 => 3), Dict()]
        layer_cfg = [Dict(:grid_len => 2), Dict(:grid_len => 2), Dict(:grid_len => 2)]
        layer_type = KDense
        model = HydroKAN.KAN(width, mult_arity, layer_cfg, layer_type)
        @test model.in_dims == 3
        @test model.out_dims == 2
        @test model.layers.layer_1.in_dims == 3
        @test model.layers.layer_1.out_dims == 8
        @test model.layers.layer_2.in_dims == 5
        @test model.layers.layer_2.out_dims == 3
        @test model.layers.layer_3.in_dims == 3
        @test model.layers.layer_3.out_dims == 2
    end

    @testset "Test whether the parameters provided by layer config are successful" begin
        width = [3, 5, 2]
        mult_arity = [Dict()]
        base_cfg_1 = Dict(:grid_len => 2, :normalizer => tanh, :basis_func => rbf, :base_act => swish, :allow_fast_activation=>false)
        base_cfg_2 = Dict(:grid_len => 3, :normalizer => sigmoid, :basis_func => rswaf, :base_act => swish, :allow_fast_activation=>false)
        layer_cfg = [base_cfg_1, base_cfg_2]
        layer_type = KDense
        model = HydroKAN.KAN(width, mult_arity, layer_cfg, layer_type)
        @test model.layers.layer_1.grid_len == 2
        @test model.layers.layer_1.normalizer == tanh
        @test model.layers.layer_1.basis_func == rbf
        @test model.layers.layer_1.base_act == swish
        @test model.layers.layer_2.grid_len == 3
        @test model.layers.layer_2.normalizer == sigmoid
        @test model.layers.layer_2.basis_func == rswaf
        @test model.layers.layer_2.base_act == swish
    end

    @testset "Test whether the KAN model calculation is correct without the introduction of multiplication operators" begin
        width = [3, 5, 2]
        mult_arity = [Dict()]
        base_cfg = Dict(:grid_len => 2, :normalizer => tanh, :basis_func => rbf, :base_act => swish)
        layer_cfg = [base_cfg, copy(base_cfg)]
        layer_type = KDense
        model_1 = HydroKAN.KAN(width, mult_arity, layer_cfg, layer_type)
        model_2 = Lux.Chain(
            KDense(3, 5, 2, normalizer=tanh, basis_func=rbf, base_act=swish),
            KDense(5, 2, 2, normalizer=tanh, basis_func=rbf, base_act=swish),
        )
        rng = StableRNG(42)
        ps1, st1 = Lux.setup(rng, model_1)
        x = rand(3, 10)
        y1, _ = model_1(x, ps1, st1)
        y2, _ = model_2(x, ps1, st1)
        @test size(y1) == size(y2)
        @test y1 ≈ y2
    end

    @testset "Test whether the KAN model calculation is correct when the multiplication operator is introduced" begin
        width = [3, 5, 2]
        mult_arity = [Dict(1 => 2, 2 => 3)]
        base_cfg_1 = Dict(:grid_len => 2, :normalizer => tanh, :basis_func => rbf, :base_act => swish)
        base_cfg_2 = Dict(:grid_len => 2, :normalizer => tanh, :basis_func => rbf, :base_act => swish)
        layer_cfg = [base_cfg_1, base_cfg_2]
        layer_type = KDense
        model = HydroKAN.KAN(width, mult_arity, layer_cfg, layer_type)
        layer_1 = KDense(3, 8, 2, normalizer=tanh, basis_func=rbf, base_act=swish)
        layer_2 = KDense(5, 2, 2, normalizer=tanh, basis_func=rbf, base_act=swish)

        rng = StableRNG(42)
        ps1, st1 = Lux.setup(rng, model)

        x = rand(3, 10)
        y1, _ = model(x, ps1, st1)
        layer_1_out, _ = layer_1(x, ps1.layer_1, st1.layer_1)
        println(size(layer_1_out))

        layer_2_in = reduce(vcat, [
            reshape(layer_1_out[1, :] .* layer_1_out[2, :], 1, :),
            reshape(layer_1_out[3, :] .* layer_1_out[4, :] .* layer_1_out[5, :], 1, :),
            layer_1_out[6:end, :]])
        layer_2_out, _ = layer_2(layer_2_in, ps1.layer_2, st1.layer_2)
        @test size(y1) == size(layer_2_out)
        @test y1 ≈ layer_2_out
    end
end