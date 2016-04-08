require 'lfs'
require 'image'

function get_train_clip(batch, video, point)
    local x = point[1]
    local y = point[2]
    local t = point[3]
    local clip = torch.Tensor(3, 60, 58, 58)
    local index = 1
    local f = io.input('/media/caffe/LZH/datasets/trainlist0' .. tostring(batch) .. '.txt')
    local c = io.read('*a')
    c = c:split('\n')
    local postfix = string.sub(c[video]:split(' ')[1], 1, -5)
    local image_path_prefix = '/media/caffe/LZH/datasets/ucf_imgs/' .. postfix
    for i = 1, 30 do
        local image_path = image_path_prefix .. '/image_' .. string.format('%04d', t - 1 + i) .. '.jpg'
        local im = image.loadJPG(image_path) 
        clip[ { {}, {index}, {}, {} } ] = im[ { {}, {y - 29, y + 29 - 1}, {x - 29, x + 29 - 1} } ]
        index = index + 1
    end
    for i = 0, 29 do
        local image_path = image_path_prefix .. '/image_' .. string.format('%04d', t - 1 - i) .. '.jpg'
        local im = image.loadJPG(image_path) 
        clip[ { {}, {index}, {}, {} } ] = im[ { {}, {y - 29, y + 29 - 1}, {x - 29, x + 29 - 1} } ]
        index = index + 1
    end
    return clip
end

function get_train_points(batch)
    local point_file = '/media/caffe/LZH/datasets/train_' .. tostring(batch) .. '_points.t7'
    local file, err = io.open(point_file)
    if err == nil then
        local t = sys.clock()
        print('point file exists')
        print('load point file ...')
        local res =  torch.load(point_file)
        print('load point file done ..,')
        print('load point file takes: ', sys.clock() - t, 's')
        return res
    else
        print('point file doesnt exists')
        local t = sys.clock()
        local f = io.input('/media/caffe/LZH/datasets/trainlist0' .. tostring(batch) .. '.txt')
        local c = io.read('*a')
        c = c:split('\n')
        local points = {}
        print('points num = ', #c, '\n')
        for i = 1, #c do
            c[i] = c[i]:split(' ')
            video_path = '/media/caffe/LZH/datasets/ucf_imgs/' .. string.sub(c[i][1], 1, -5)
            frame_num = get_frame_num(video_path)
            point = get_rand_points(frame_num)
--            print(i, '-th point size: ', #point)
            table.insert(points, point)
        end
        print('saving train_' .. tostring(batch) .. ' point file')
        torch.save('/media/caffe/LZH/datasets/train_' .. tostring(batch) .. '_points.t7', points)
        print('saving done')
        print('load point file takes: ', sys.clock() - t, 's')
        return points
    end
end

function get_train_video_num(batch)
    local f = io.input('/media/caffe/LZH/datasets/trainlist0' .. tostring(batch) .. '.txt')
    local c = io.read('*a')
    c = c:split('\n')
    return #c
end

function get_train_label(batch, video)
    local f = io.input('/media/caffe/LZH/datasets/trainlist0' .. tostring(batch) .. '.txt')
    local c = io.read('*a')
    local v  = c:split('\n')[video]:split(' ')
    return v[2]
end

function get_frame_num(video_path)
    local res = 0
    ret, files, iter = pcall(lfs.dir, video_path) if ret == true then
        for f in files, iter do
            res = res + 1
        end
    else
        print(video_path ' has no images ...')
        return res
    end
    return res - 2
end

function get_rand_points(frame_num)
    local points = {}
    local w_s = 30
    local w_e = 320 - 30
    local h_s = 30
    local h_e = 240 - 30
    local t_s = 30
    local t_e = frame_num - 30
    local t_stride = 4
    for t = t_s, t_e, t_stride do
        for w = 1, 4 do
            x = math.random(w_s, w_e)
            for h = 1, 4 do
                y = math.random(h_s, h_e)
                local point = {x, y, t} 
                table.insert(points, point)
            end
        end
    end
    return points
end
