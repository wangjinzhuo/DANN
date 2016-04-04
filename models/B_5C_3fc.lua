require 'nn'

    local input = torch.Tensor(3, 60, 58, 58)
    model = nn.Sequential()


    --conv1
    model:add( nn.VolumetricConvolution(3, 64, 3, 3, 3, 1, 1, 1, 1, 1, 1) ) -- input/output, filer size, stride, padding
    model:add( nn.VolumetricMaxPooling(1, 2, 2, 1, 2, 2) ) -- filter size, padding
    model:add( nn.ReLU(true) )

    --conv2
    model:add( nn.VolumetricConvolution(64, 128, 3, 3, 3, 1, 1, 1, 1, 1, 1) )
    model:add( nn.VolumetricMaxPooling(2, 2, 2, 2, 2, 2) )
    model:add( nn.ReLU(true) )

    --conv3
    model:add( nn.VolumetricConvolution(128, 256, 3, 3, 3, 1, 1, 1, 1, 1, 1) )
    model:add( nn.VolumetricMaxPooling(2, 2, 2, 2, 2, 2) )
    model:add( nn.ReLU(true) )

    --conv4
    model:add( nn.VolumetricConvolution(256, 256, 3, 3, 3, 1, 1, 1, 1, 1, 1) )
    model:add( nn.VolumetricMaxPooling(2, 2, 2, 2, 2, 2) )
    model:add( nn.ReLU(true) )

    --conv5
    model:add( nn.VolumetricConvolution(256, 256, 3, 3, 3, 1, 1, 1, 1, 1, 1) ) 
    model:add( nn.VolumetricMaxPooling(2, 2, 2, 2, 2, 2) )
    model:add( nn.ReLU(true) )

    model:add( nn.View(768) )

    --fc6
    model:add( nn.Linear(768, 2048) )
    model:add( nn.ReLU(true) )
    model:add( nn.Dropout(0.90000) )

    --fc7
    model:add( nn.Linear(2048, 2048) )
    model:add( nn.ReLU(true) )
    model:add( nn.Dropout(0.90000) )

    --fc8
    model:add( nn.Linear(2048, 101) )
    model:add( nn.SoftMax() )


    --[[
    input:cuda()
    model:cuda()
    output = model:forward(input)
    print(#output)
    ]]
