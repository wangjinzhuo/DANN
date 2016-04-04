require 'nn'
require 'cunn'
require 'rnn'
require 'models/RCLayer'
require 'models/RCLMaxPooling'

local input = torch.rand(3, 60, 58, 58)
model = nn.Sequential()


--conv1
model:add( nn.VolumetricConvolution(3, 64, 3, 3, 3, 1, 1, 1, 1, 1, 1) ) -- input/output, filer size, stride, padding
rnn1 = nn.RCLayer(
    nn.Recurrent(
        nn.Add(64, true), nn.VolumetricConvolution(64, 64, 3, 3, 3, 1, 1, 1, 1, 1, 1),
        nn.VolumetricConvolution(64, 64, 3, 3, 3, 1, 1, 1, 1, 1, 1), nn.ReLU(true),
        6
    ),
    6
)
model:add(rnn1)
model:add( nn.RCLMaxPooling(1, 2, 2, 1, 2, 2) ) -- filter size, padding

--conv2
model:add( nn.VolumetricConvolution(64, 128, 3, 3, 3, 1, 1, 1, 1, 1, 1) )
rnn2 = nn.RCLayer(
    nn.Recurrent(
        nn.Add(128, true), nn.VolumetricConvolution(128, 128, 3, 3, 3, 1, 1, 1, 1, 1, 1),
        nn.VolumetricConvolution(128, 128, 3, 3, 3, 1, 1, 1, 1, 1, 1), nn.ReLU(true),
        6
    ),
    6
)
model:add(rnn2)
model:add( nn.RCLMaxPooling(2, 2, 2, 2, 2, 2) )

--conv3
model:add( nn.VolumetricConvolution(128, 256, 3, 3, 3, 1, 1, 1, 1, 1, 1) )
rnn3 = nn.RCLayer(
    nn.Recurrent(
        nn.Add(256, true), nn.VolumetricConvolution(256, 256, 3, 3, 3, 1, 1, 1, 1, 1, 1),
        nn.VolumetricConvolution(256, 256, 3, 3, 3, 1, 1, 1, 1, 1, 1), nn.ReLU(true),
        6
    ),
    6
)
model:add(rnn3)
model:add( nn.RCLMaxPooling(2, 2, 2, 2, 2, 2) )

--conv4
model:add( nn.VolumetricConvolution(256, 256, 3, 3, 3, 1, 1, 1, 1, 1, 1) )
rnn4 = nn.RCLayer(
    nn.Recurrent(
        nn.Add(256, true), nn.VolumetricConvolution(256, 256, 3, 3, 3, 1, 1, 1, 1, 1, 1),
        nn.VolumetricConvolution(256, 256, 3, 3, 3, 1, 1, 1, 1, 1, 1), nn.ReLU(true),
        6
    ),
    6
)
model:add(rnn4)
model:add( nn.RCLMaxPooling(2, 2, 2, 2, 2, 2) )

--conv5
model:add( nn.VolumetricConvolution(256, 256, 3, 3, 3, 1, 1, 1, 1, 1, 1) ) 
rnn5 = nn.RCLayer(
    nn.Recurrent(
        nn.Add(256, true), nn.VolumetricConvolution(256, 256, 3, 3, 3, 1, 1, 1, 1, 1, 1),
        nn.VolumetricConvolution(256, 256, 3, 3, 3, 1, 1, 1, 1, 1, 1), nn.ReLU(true),
        6
    ),
    6
)
model:add(rnn5)
model:add( nn.RCLMaxPooling(2, 2, 2, 2, 2, 2) )

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



