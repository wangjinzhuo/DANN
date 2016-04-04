train_1 = '/medai/cnn/My Passport/datasets/trainlist01.txt' -- 9538
train_2 = '/medai/cnn/My Passport/datasets/trainlist02.txt' -- 9587
train_3 = '/medai/cnn/My Passport/datasets/trainlist03.txt' -- 9625

test_1 = '/medai/cnn/My Passport/datasets/testlist01.txt' -- 3784
test_2 = '/medai/cnn/My Passport/datasets/testlist02.txt' -- 3735
test_3 = '/medai/cnn/My Passport/datasets/testlist03.txt' -- 3697

imagePath = '/media/cnn/My Passport/datasets/ucf_imags'

function getFrameNumGivenVideoPath(videoPath)
    local frameNum = 0
    ret, files, iter = pcall(lfs.dir, videoPath)
    for f in files, iter do
        frameNum = frameNum + 1
    end
    return frameNum - 2
end


function getPointsGivenNum(frameNum)
    local t = sys.clock()
    local points = {} 
    local widthStart  = 30 
    local widthEnd = 320 - 30
    local heightStart = 30
    local heightEnd = 240 - 30
    local frameStart = 30
    local frameEnd = frameNum - 30 
    local frameStride = 4
    for frame = frameStart, frameEnd, frameStride do
        for ww = 1, 4 do
            x = math.random(widthStart, widthEnd)
            for hh = 1, 4 do
                y = math.random(heightStart, heightEnd)
                local point = {x, y, frame}       
                table.insert(points, point)
            end
        end
    end
   -- print('randGenerateSamplingPoint Time: ', sys.clock() - t)
    return points -- return all the sampling points given a video containing frameNum frames
end

function getVideoNum(filePath)
    f = io.input(filePath)
    content = io.read('*a'):split('\n')
    if string.sub(filePath, 1, 3) == 'tra' then
        for i = 1, #content do
            content[i] = content[i]:split(' ')
        end
    end
    local allImagePath = {}
    for i = 1, #content do
        videoPath = imagePath .. '/' .. string.sub(content[i], 1, -5)
        table.insert(allImagePath, videoPath)
    end
    return #content, allImagePath
end

function getPoints(filePath)
    local allPoints = {}
    local videoNum, image_path = getVideoNum(filePath)
    for i = 1, videoNum do
        local points = getPointsGivenVideo(allVideoPathes[i])
        table.insert(allPoints, points)
    end 
    return allPoints, allImagePath
end

tr_1_points, tr_1_image_path = getPoints(train_1)
tr_2_points, tr_2_image_path = getPoints(train_2)
tr_3_points, tr_3_image_path = getPoints(train_3)
tr_points = {tr_1_points, tr_2_points, tr_3_points}
tr_image_path = {tr_1_image_path, tr_2_image_path, tr_3_image_path}

te_1_points, tr_1_image_path = getPoints(test_1)
te_2_points, tr_2_image_path = getPoints(test_2)
te_3_points, tr_3_image_path = getPoints(test_3)
te_points = {tr_1_points, tr_2_points, tr_3_points}
te_image_path = {te_1_image_path, te_2_image_path, te_3_image_path}

function getClip(datatype, batch, video, pointNum)
    if datatype == 'tr' then
        points = tr_points[batch]
        pathes = tr_image_path[batch]
    else
        points = te_points[batch]
        pathes = te_image_path[batch]
    end
    center = points[pointNum]
    local clip = torch.Tensor(3, 60, 58, 58)
    local index = 1
    for t = 1, 30 do
        local im = image.loadJPG(pathes[video] .. '/' .. tostring(center[3] + t) .. '.jpg')
        clip[{ {}, {index}, {}, {} }] = im[{ {}, {center[2] - 29, center[2] + 29 - 1}, {center[1] - 29, center[1] + 29 - 1} }]
        index = index + 1
    end
    for t = 0, 29 do
        local im = image.loadJPG(pathes[video] .. '/' .. tostring(center[3] - t) .. '.jpg')
        clip[{ {}, {index}, {}, {} }] = im[{ {}, {center[2] - 29, center[2] + 29 - 1}, {center[1] - 29, center[1] + 29 - 1} }]
        index = index + 1
    end
    return clip
end
