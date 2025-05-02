module module_neural_net
    use netcdf
    use shr_kind_mod,   only: r8=>shr_kind_r8
    !use shr_kind_mod,   only: r4=>shr_kind_r4
    use ppgrid,          only:  pver
    ! AGS
    use ftorch,         only: torch_kCPU, torch_tensor, torch_model, torch_tensor_from_array
    use ftorch,         only: torch_model_load, torch_model_forward, torch_tensor_delete
    implicit none
    type Dense
        integer :: input_size
        integer :: output_size
        integer :: batch_size
        integer :: activation
        real(kind=r8), allocatable :: weights(:, :)
        real(kind=r8), allocatable :: bias(:)
    end type Dense

    type DenseData
        real(kind=r8), allocatable :: input(:, :)
        real(kind=r8), allocatable :: output(:, :)
    end type DenseData

contains

    subroutine apply_dense(input, layer, output)
        ! Description: Pass a set of input data through a single dense layer and nonlinear activation function
        !
        ! Inputs:
        ! layer (input): a single Dense object
        ! input (input): a 2D array where the rows are different examples and
        !   the columns are different model inputs
        !
        ! Output:
        ! output: output of the dense layer as a 2D array with shape (number of inputs, number of neurons)
        real(kind=r8), dimension(:, :), intent(in) :: input
        type(Dense), intent(in) :: layer
        real(kind=r8), dimension(size(input, 1), layer%output_size), intent(out) :: output
        real(kind=r8), dimension(size(input, 1), layer%output_size) :: dense_output
        integer :: i, j, num_examples
        real(kind=r8) :: alpha, beta
        external :: dgemm
        alpha = 1_r8
        beta = 1_r8
        dense_output = 0_r8
        output = 0_r8
        num_examples = size(input, 1)
        call dgemm('n', 'n', num_examples, layer%output_size, layer%input_size, &
            alpha, input, num_examples, layer%weights, layer%input_size, beta, dense_output, num_examples)
        do i=1, num_examples
            do j=1, layer%output_size
                dense_output(i, j) = dense_output(i, j) + layer%bias(j)
            end do
        end do
        call apply_activation(dense_output, layer%activation, output)
    end subroutine apply_dense

    subroutine apply_activation(input, activation_type, output)
        ! Description: Apply a nonlinear activation function to a given array of input values.
        !
        ! Inputs:
        ! input: A 2D array
        ! activation_type: string describing which activation is being applied. If the activation
        !       type does not match any of the available options, the linear activation is applied.
        !       Currently supported activations are:
        !           relu
        !           elu
        !           selu
        !           sigmoid
        !           tanh
        !           softmax
        !           linear
        ! Output:
        ! output: Array of the same dimensions as input with the nonlinear activation applied.
        real(kind=r8), dimension(:, :), intent(in) :: input
        integer, intent(in) :: activation_type
        real(kind=r8), dimension(size(input, 1), size(input, 2)), intent(out) :: output

        real(kind=r8), dimension(size(input, 1)) :: softmax_sum
        real(kind=r8), parameter :: selu_alpha = 1.6732_r8
        real(kind=r8), parameter :: selu_lambda = 1.0507_r8
        real(kind=r8), parameter :: zero = 0.0_r8
        integer :: i, j
        select case (activation_type)
            case (0)
                output = input
            case (1)
                do i=1,size(input, 1)
                    do j=1, size(input,2)
                        output(i, j) = dmax1(input(i, j), zero)
                    end do
                end do
            case (2)
                output = 1.0_r8 / (1.0_r8 + dexp(-input))
            case (3)
                do i=1,size(input, 1)
                    do j=1, size(input,2)
                        if (input(i, j) >= 0) then
                            output(i, j) = input(i, j)
                        else
                            output(i, j) = dexp(input(i, j))-1.0_r8
                        end if
                    end do
                end do
            case (4)
                do i=1,size(input, 1)
                    do j=1, size(input,2)
                        if (input(i, j) >= 0) then
                            output(i, j) = input(i, j)
                        else
                            output(i, j) = selu_lambda * ( selu_alpha * dexp(input(i, j)) - selu_alpha)
                        end if
                    end do
                end do
            case (5)
                output = tanh(input)
            case (6)
                softmax_sum = sum(dexp(input), dim=2)
                do i=1, size(input, 1)
                    do j=1, size(input, 2)
                        output(i, j) = dexp(input(i, j)) / softmax_sum(i)
                    end do
                end do
            case default
                output = input
        end select
    end subroutine apply_activation

    subroutine init_neural_net(filename, batch_size, model_ftorch, iulog, errstring)
        ! init_neural_net
        ! Description: Loads neural network model using FTorch interface from a file
        ! 
        ! Input:
        ! filename: Full path to the model file
        ! batch_size: number of items in single batch (for potential batch inference)
        !
        ! Output:
        ! model_ftorch: FTorch model handle
        !
        use ftorch, only: torch_model_load, torch_kCPU
    
        character(len=*), intent(in) :: filename
        integer, intent(in) :: batch_size
    
        type(torch_model), intent(out) :: model_ftorch
        integer, intent(in) :: iulog
        character(128), intent(out) :: errstring  ! output status (non-blank for error return)
    
        ! Load the model using FTorch interface
          call torch_model_load(model_ftorch, trim(filename), torch_kCPU)
        ! call torch_model_load(model_ftorch, trim(filename))   
        ! Log successful model loading
        write(iulog,*) 'Successfully loaded neural network model from: ', trim(filename)
        write(iulog,*) 'Model batch size set to: ', batch_size
        
    end subroutine init_neural_net

    !subroutine init_neural_net(filename, batch_size, neural_net_model, iulog, errstring)
        ! init_neuralnet
        ! Description: Loads dense neural network weights from a netCDF file and builds an array of
        ! Dense types from the weights and activations.
        !
        ! Input:
        ! filename: Full path to the netCDF file
        ! batch_size: number of items in single batch. Used to set intermediate array sizes.
        !
        ! Output:
        ! neural_net_model (output): array of Dense layers composing a densely connected neural network
        !
        !character(len=*), intent(in) :: filename
        !integer, intent(in) :: batch_size
        ! AGS
        ! type(torch_model), allocatable, intent(out) :: neural_net_model(:)
        ! type(Dense), allocatable, intent(out) :: neural_net_model(:)
        ! integer,          intent(in)  :: iulog
        ! character(128),   intent(out) :: errstring  ! output status (non-blank for error return)

        !integer :: num_layers
        ! AGS 
        !num_layers = pver

        !allocate(neural_net_model(1:num_layers))
        ! AGS
        ! call torch_model_load(neural_net_model, filename, torch_kCPU)


    !end subroutine init_neural_net

    subroutine load_quantile_scale_values(filename, scale_values, iulog, errstring)
        character(len = *), intent(in) :: filename
        real(kind = r8), allocatable, intent(out) :: scale_values(:, :)
        integer,          intent(in)  :: iulog
        character(128),   intent(out) :: errstring  ! output status (non-blank for error return)

        real(kind = r8), allocatable :: temp_scale_values(:, :)
        character(len=8) :: quantile_dim_name = "quantile"
        character(len=7) :: column_dim_name = "column"
        character(len=9) :: ref_var_name = "reference"
        character(len=9) :: quant_var_name = "quantiles"
        integer :: ncid, quantile_id, column_id, quantile_dim, column_dim, ref_var_id, quant_var_id
        
        errstring = ''

        call check(nf90_open(filename, nf90_nowrite, ncid),errstring)
        if (trim(errstring) /= '') return
        call check(nf90_inq_dimid(ncid, quantile_dim_name, quantile_id),errstring)
        if (trim(errstring) /= '') return
        call check(nf90_inq_dimid(ncid, column_dim_name, column_id),errstring)
        if (trim(errstring) /= '') return
        call check(nf90_inquire_dimension(ncid, quantile_id, &
                quantile_dim_name, quantile_dim),errstring)
        if (trim(errstring) /= '') return
        call check(nf90_inquire_dimension(ncid, column_id, &
                column_dim_name, column_dim),errstring)
        if (trim(errstring) /= '') return
        allocate(scale_values(quantile_dim, column_dim + 1))
        allocate(temp_scale_values(column_dim + 1, quantile_dim))
        call check(nf90_inq_varid(ncid, ref_var_name, ref_var_id),errstring)
        if (trim(errstring) /= '') return
        write(iulog,*) "load ref var"
        call check(nf90_get_var(ncid, ref_var_id, temp_scale_values(1, :)),errstring)
        if (trim(errstring) /= '') return
        call check(nf90_inq_varid(ncid, quant_var_name, quant_var_id),errstring)
        if (trim(errstring) /= '') return
        write(iulog,*) "load quant var"
        call check(nf90_get_var(ncid, quant_var_id, temp_scale_values(2:column_dim + 1, :)),errstring)
        if (trim(errstring) /= '') return
        scale_values = transpose(temp_scale_values)
        call check(nf90_close(ncid),errstring)
        if (trim(errstring) /= '') return
    end subroutine load_quantile_scale_values

    subroutine linear_interp_forward(x_in, xs, ys, y_in)
        real(kind = r8), dimension(:), intent(in) :: x_in
        real(kind = r8), dimension(:), intent(in) :: xs
        real(kind = r8), dimension(:), intent(in) :: ys
        real(kind = r8), dimension(size(x_in, 1)), intent(out) :: y_in
        integer :: i, j, jl, jr, x_in_size, xs_size, x_pos
        real(kind = r8) :: slope
        x_in_size = size(x_in)
        xs_size = size(xs)
        do i = 1, x_in_size
            call binary_search_left(xs, x_in(i), jl)
            call binary_search_right(xs, x_in(i), jr)
            j = (jl + jr) / 2
            if ((j == 1) .or. (j == xs_size) .or. (xs(j + 1) - xs(j) == 0)) then
                y_in(i) = ys(j)
            else
                slope = (ys(j + 1) - ys(j)) / (xs(j + 1) - xs(j))
                y_in(i) = slope * (x_in(i) - xs(j)) + ys(j) 
            end if
        end do
    end subroutine linear_interp_forward

    subroutine linear_interp_inverse(x_in, xs, ys, y_in)
        real(kind = r8), dimension(:), intent(in) :: x_in
        real(kind = r8), dimension(:), intent(in) :: xs
        real(kind = r8), dimension(:), intent(in) :: ys
        real(kind = r8), dimension(size(x_in, 1)), intent(out) :: y_in
        integer :: i, j, x_in_size, xs_size, x_pos
        real(kind = r8) :: slope
        x_in_size = size(x_in)
        xs_size = size(xs)
        do i = 1, x_in_size
            call binary_search_left(xs, x_in(i), j)
            if ((j == 1) .or. (j == xs_size) .or. (xs(j + 1) - xs(j) == 0)) then
                y_in(i) = ys(j)
            else
                slope = (ys(j + 1) - ys(j)) / (xs(j + 1) - xs(j))
                y_in(i) = slope * (x_in(i) - xs(j)) + ys(j) 
            end if
        end do
    end subroutine linear_interp_inverse

    subroutine binary_search_left(x, target_val, val_index)
        ! binary_search_left finds the leftmost index to insert
        ! the target value in a sorted array x and returns
        ! that index in val_index.
        real(kind = r8), dimension(:), intent(in) :: x
        real(kind = r8), intent(in) :: target_val
        integer, intent(out) :: val_index
        integer :: min_idx, max_idx, mid_idx
        real(kind = r8) :: mid_val
        min_idx = 1
        max_idx = size(x)
        mid_val = 1
        mid_idx = 1
        if (target_val <= x(min_idx)) then
            max_idx = min_idx
        end if
        do while (min_idx < max_idx)
            mid_idx = min_idx + (max_idx - min_idx) / 2
            mid_val = x(mid_idx)
            if (mid_val < target_val) then
                min_idx = mid_idx + 1
            else if (mid_val >= target_val) then
                max_idx = mid_idx - 1
            end if
        end do
        val_index = min_idx
    end subroutine binary_search_left

    subroutine binary_search_right(x, target_val, val_index)
        ! binary_search_right finds the rightmost index to insert
        ! the target value in a sorted array x and returns
        ! that index in val_index.
        real(kind = r8), dimension(:), intent(in) :: x
        real(kind = r8), intent(in) :: target_val
        integer, intent(out) :: val_index
        integer :: min_idx, max_idx, mid_idx
        real(kind = r8) :: mid_val
        min_idx = 1
        max_idx = size(x)
        mid_idx = 1
        if (target_val >= x(max_idx)) then
            min_idx = max_idx
        end if
        do while (min_idx < max_idx)
            mid_idx = min_idx + (max_idx - min_idx) / 2
            mid_val = x(mid_idx)
            if (mid_val <= target_val) then
                min_idx = mid_idx + 1
            else if (mid_val > target_val) then
                max_idx = mid_idx - 1
            end if
        end do
        val_index = max_idx
    end subroutine binary_search_right

    subroutine quantile_transform(x_inputs, scale_values, x_transformed)
        real(kind = r8), dimension(:, :), intent(in) :: x_inputs
        real(kind = r8), dimension(:, :), intent(in) :: scale_values
        real(kind = r8), dimension(size(x_inputs, 1), size(x_inputs, 2)), intent(out) :: x_transformed
        integer :: j, x_size, scale_size
        x_size = size(x_inputs, 1)
        scale_size = size(scale_values, 1)
        do j = 1, size(x_inputs, 2)
            call linear_interp_forward(x_inputs(:, j), scale_values(:, j + 1), &
                    scale_values(:, 1), x_transformed(:, j))
        end do
    end subroutine quantile_transform

    subroutine quantile_inv_transform(x_inputs, scale_values, x_transformed)
        real(kind = r8), dimension(:, :), intent(in) :: x_inputs
        real(kind = r8), dimension(:, :), intent(in) :: scale_values
        real(kind = r8), dimension(size(x_inputs, 1), size(x_inputs, 2)), intent(out) :: x_transformed
        integer :: j, x_size, scale_size
        x_size = size(x_inputs, 1)
        scale_size = size(scale_values, 1)
        do j = 1, size(x_inputs, 2)
            call linear_interp_inverse(x_inputs(:, j), scale_values(:, 1), scale_values(:, j + 1), x_transformed(:, j))
        end do
    end subroutine quantile_inv_transform

    subroutine neural_net_predict(input, model_ftorch, prediction, iulog)
        ! neural_net_predict
        ! Description: generate prediction from neural network model using FTorch interface
        !
        ! Args:
        ! input (input): 2D array of input values. Each row is a separate instance and each column is a model input.
        ! model_ftorch (input): FTorch model loaded by init_neural_net
        ! prediction (output): The prediction of the neural network as a 2D array
        ! iulog (input): Log file unit number

        use ftorch, only: torch_tensor, torch_tensor_from_array, torch_model_forward
        use ftorch, only: torch_kCPU
    
        real(kind=r8), intent(in) :: input(:, :)
        type(torch_model), intent(in) :: model_ftorch
        real(kind=r8), intent(out) :: prediction(:, :)
        integer, intent(in) :: iulog
    
        ! Local variables
        integer :: i, batch_size, num_samples, num_features, num_outputs
        integer :: current_batch_size
    
        ! FTorch tensors
        type(torch_tensor), dimension(1) :: in_tensor, out_tensor
    
        ! Tensor shapes and layouts
        integer :: in_layout(2), out_layout(2)
    
        ! Data arrays for tensor conversion
        real(r8), allocatable, target :: in_data_single(:,:)
        real(r8), allocatable, target :: out_data_single(:,:)

        ! Get dimensions
        num_samples = size(input, 1)
        num_features = size(input, 2)
        num_outputs = size(prediction, 2)
        batch_size = min(64, num_samples)  ! Default batch size or use parameter
    
        ! Set up tensor layouts
        in_layout = [1, 2]  ! [row, col] layout
        out_layout = [1, 2]

        ! Process batches
        do i = 1, num_samples, batch_size
            ! Calculate current batch size (might be smaller for final batch)
            current_batch_size = min(batch_size, num_samples - i + 1)
        
            ! Allocate temporary arrays for current batch
            allocate(in_data_single(current_batch_size, num_features))
            allocate(out_data_single(current_batch_size, num_outputs))
        
            ! Convert input batch to single precision
            in_data_single = real(input(i:i+current_batch_size-1, :), r8)
        
            ! Create tensors from arrays
            call torch_tensor_from_array(in_tensor(1), in_data_single, in_layout, torch_kCPU)
            call torch_tensor_from_array(out_tensor(1), out_data_single, out_layout, torch_kCPU)
        
            ! Forward pass through the model
            call torch_model_forward(model_ftorch, in_tensor, out_tensor)
        
            ! Convert output to double precision and store in prediction array
            prediction(i:i+current_batch_size-1, :) = real(out_data_single, r8)
        
            ! Clean up temporary arrays
            deallocate(in_data_single)
            deallocate(out_data_single)
        end do

        ! Apply sigmoid function if needed for probability output
        ! (Uncomment if the model output needs sigmoid activation)
        ! prediction(:,:) = 1.0_r8 / (1.0_r8 + exp(-prediction(:,:)))
    
        write(iulog,*) 'Neural network prediction completed for', num_samples, 'samples'
    
    end subroutine neural_net_predict
    
    
    !subroutine neural_net_predict(input, neural_net_model, prediction, iulog)
        ! neural_net_predict
        ! Description: generate prediction from neural network model for an arbitrary set of input values
        !
        ! Args:
        ! input (input): 2D array of input values. Each row is a separate instance and each column is a model input.
        ! neural_net_model (input): Array of type(Dense) objects
        ! prediction (output): The prediction of the neural network as a 2D array of dimension (examples, outputs)
        ! real(kind=r8), intent(in) :: input(:, :)
        ! type(Dense), intent(in) :: neural_net_model(:)
        ! real(kind=r8), intent(out) :: prediction(size(input, 1), neural_net_model(size(neural_net_model))%output_size)
        ! integer :: bi, i, j, num_layers
        ! integer :: batch_size
        ! integer :: input_size
        ! integer :: batch_index_size
        ! integer, allocatable :: batch_indices(:)
        ! integer,  intent(in)  :: iulog
        ! type(DenseData) :: neural_net_data(size(neural_net_model))
        ! input_size = size(input, 1)
        ! num_layers = size(neural_net_model)
        ! batch_size = neural_net_model(1)%batch_size
        ! batch_index_size = input_size / batch_size
        ! allocate(batch_indices(batch_index_size))
        ! i = 1
        ! do bi=batch_size, input_size, batch_size
        !    batch_indices(i) = bi
        !    i = i + 1
        !end do
        !do j=1, num_layers
        !    allocate(neural_net_data(j)%input(batch_size, neural_net_model(j)%input_size))
        !    allocate(neural_net_data(j)%output(batch_size, neural_net_model(j)%output_size))
        !end do
        !batch_indices(batch_index_size) = input_size
        !do bi=1, batch_index_size
        !    neural_net_data(1)%input = input(batch_indices(bi)-batch_size+1:batch_indices(bi), :)
        !    do i=1, num_layers - 1
        !        call apply_dense(neural_net_data(i)%input, neural_net_model(i), neural_net_data(i)%output)
        !        neural_net_data(i + 1)%input = neural_net_data(i)%output
        !    end do
        !    call apply_dense(neural_net_data(num_layers)%input, neural_net_model(num_layers), &
        !                     neural_net_data(num_layers)%output)
        !    prediction(batch_indices(bi)-batch_size + 1:batch_indices(bi), :) = &
        !            neural_net_data(num_layers)%output
        !end do
        !do j=1, num_layers
        !    deallocate(neural_net_data(j)%input)
        !   deallocate(neural_net_data(j)%output)
        !end do
        !deallocate(batch_indices)
    !end subroutine neural_net_predict

    subroutine standard_scaler_transform(input_data, scale_values, transformed_data, errstring)
        ! Perform z-score normalization of input_data table. Equivalent to scikit-learn StandardScaler.
        !
        ! Inputs:
        !   input_data: 2D array where rows are examples and columns are variables
        !   scale_values: 2D array where rows are the input variables and columns are mean and standard deviation
        ! Output:
        !   transformed_data: 2D array with the same shape as input_data containing the transformed values.
        real(r8), intent(in) :: input_data(:, :)
        real(r8), intent(in) :: scale_values(:, :)
        real(r8), intent(out) :: transformed_data(size(input_data, 1), size(input_data, 2))
        character(128),   intent(out) :: errstring  ! output status (non-blank for error return)
        integer :: i
 
        errstring = ''
        if (size(input_data, 2) /= size(scale_values, 1)) then
            write(errstring,*) "Size mismatch between input data and scale values", size(input_data, 2), size(scale_values, 1)
            return
        end if
        do i=1, size(input_data, 2)
            transformed_data(:, i) = (input_data(:, i) - scale_values(i, 1)) / scale_values(i, 2)
        end do
    end subroutine standard_scaler_transform

    subroutine load_scale_values(filename, num_inputs, scale_values)
        character(len=*), intent(in) :: filename
        integer, intent(in) :: num_inputs
        real(r8), intent(out) :: scale_values(num_inputs, 2)
        character(len=40) :: row_name
        integer :: isu, i
        isu = 2
        open(isu, file=filename, access="sequential", form="formatted")
        read(isu, "(A)")
        do i=1, num_inputs
            read(isu, *) row_name, scale_values(i, 1), scale_values(i, 2)
        end do
        close(isu)
    end subroutine load_scale_values


    subroutine standard_scaler_inverse_transform(input_data, scale_values, transformed_data, errstring)
        ! Perform inverse z-score normalization of input_data table. Equivalent to scikit-learn StandardScaler.
        !
        ! Inputs:
        !   input_data: 2D array where rows are examples and columns are variables
        !   scale_values: 2D array where rows are the input variables and columns are mean and standard deviation
        ! Output:
        !   transformed_data: 2D array with the same shape as input_data containing the transformed values.
        real(r8), intent(in) :: input_data(:, :)
        real(r8), intent(in) :: scale_values(:, :)
        real(r8), intent(out) :: transformed_data(size(input_data, 1), size(input_data, 2))
        character(128),   intent(out) :: errstring  ! output status (non-blank for error return)
        integer :: i
        if (size(input_data, 2) /= size(scale_values, 1)) then
            write(errstring,*) "Size mismatch between input data and scale values", size(input_data, 2), size(scale_values, 1)
            return
        end if
        do i=1, size(input_data, 2)
            transformed_data(:, i) = input_data(:, i) * scale_values(i, 2) + scale_values(i, 1)
        end do
    end subroutine standard_scaler_inverse_transform

    subroutine minmax_scaler_transform(input_data, scale_values, transformed_data, errstring)
        ! Perform min-max scaling of input_data table. Equivalent to scikit-learn MinMaxScaler.
        !
        ! Inputs:
        !   input_data: 2D array where rows are examples and columns are variables
        !   scale_values: 2D array where rows are the input variables and columns are min and max.
        ! Output:
        !   transformed_data: 2D array with the same shape as input_data containing the transformed values.
        real(r8), intent(in) :: input_data(:, :)
        real(r8), intent(in) :: scale_values(:, :)
        real(r8), intent(out) :: transformed_data(size(input_data, 1), size(input_data, 2))
        character(128),   intent(out) :: errstring  ! output status (non-blank for error return)

        integer :: i
        if (size(input_data, 2) /= size(scale_values, 1)) then
            write(errstring,*) "Size mismatch between input data and scale values", size(input_data, 2), size(scale_values, 1)
            return
        end if
        do i=1, size(input_data, 2)
            transformed_data(:, i) = (input_data(:, i) - scale_values(i, 1)) / (scale_values(i, 2) - scale_values(i ,1))
        end do
    end subroutine minmax_scaler_transform

    subroutine minmax_scaler_inverse_transform(input_data, scale_values, transformed_data, errstring)
        ! Perform inverse min-max scaling of input_data table. Equivalent to scikit-learn MinMaxScaler.
        !
        ! Inputs:
        !   input_data: 2D array where rows are examples and columns are variables
        !   scale_values: 2D array where rows are the input variables and columns are min and max.
        ! Output:
        !   transformed_data: 2D array with the same shape as input_data containing the transformed values.
        real(r8), intent(in) :: input_data(:, :)
        real(r8), intent(in) :: scale_values(:, :)
        real(r8), intent(out) :: transformed_data(size(input_data, 1), size(input_data, 2))
        character(128),   intent(out) :: errstring  ! output status (non-blank for error return)

        integer :: i
        if (size(input_data, 2) /= size(scale_values, 1)) then
            write(errstring,*) "Size mismatch between input data and scale values", size(input_data, 2), size(scale_values, 1)
            return
        end if
        do i=1, size(input_data, 2)
            transformed_data(:, i) = input_data(:, i) * (scale_values(i, 2) - scale_values(i ,1)) + scale_values(i, 1)
        end do
    end subroutine minmax_scaler_inverse_transform

    subroutine check(status, errstring)
        ! Check for netCDF errors
        integer, intent ( in) :: status
        character(128),   intent(out) :: errstring  ! output status (non-blank for error return)

        errstring = ''
        if(status /= nf90_noerr) then
          errstring = trim(nf90_strerror(status))
        end if
    end subroutine check

end module module_neural_net
