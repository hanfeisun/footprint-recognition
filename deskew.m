function result = deskew(path)
    image = imread(path);
    bw = imbinarize(image);
    hull = bwconvhull(1-bw);
    cannied = edge(hull, 'canny');
    figure, imshow(image), hold on
    [H,T,R] = hough(cannied);
    P  = houghpeaks(H,5,'threshold',ceil(0.3*max(H(:))));
    lines = houghlines(cannied,T,R,P,'FillGap',5,'MinLength',7);
    % figure, imshow(image), hold on
    max_len = 0;
    for k = 1:length(lines)
       xy = [lines(k).point1; lines(k).point2];
        
       plot(xy(:,1),xy(:,2),'LineWidth',2,'Color','green');

       % Plot beginnings and ends of lines
       plot(xy(1,1),xy(1,2),'x','LineWidth',2,'Color','yellow');
       plot(xy(2,1),xy(2,2),'x','LineWidth',2,'Color','red');
    for k = 1:length(lines)
        theta = lines(k).theta;
        if theta == 0
            continue
        end
        result=255-imrotate(255-image, theta);
        smooth = imbinarize(imgaussfilt(result,0.8), 0.6);
        imwrite(smooth,strcat('deskew', path));
        break
    end
    end
