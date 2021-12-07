import os
import copy
import timeit
import logging

import cv2
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as et

from openvino.inference_engine import IECore

MODEL_PATH = 'model/FP16/'
DEVICE = 'CPU'
logging.basicConfig(level=logging.DEBUG,format='[%(levelname)s] %(message)s',handlers=[logging.FileHandler('log.txt','w','utf-8')])

def splitFileName(file):
    dirname, filename = os.path.split(file)
    basename, extname = os.path.splitext(filename)
    return dirname, basename, extname

def readXML(model):
    dname, bname, ename = splitFileName(model)
    tree = et.parse(os.path.join(dname, bname+'.xml'))
    return tree

def readBIN(model):
    dname, bname, ename = splitFileName(model)
    with open(os.path.join(dname, bname+'.bin'), 'rb') as f:
        weight = f.read()
    return weight

def findNodeFromXML(xmltree, nodeid):
    root = xmltree.getroot()
    layers = root.find('layers')
    for layer in layers.findall('layer'):
        if int(layer.attrib['id']) == nodeid:
            return layer
    return None

def modifyXMLForFeatureVectorProbing(xmltree, nodeid):
    xmlcopy = copy.deepcopy(xmltree)
    layer = findNodeFromXML(xmlcopy, nodeid)

    # obtain output port information of the target node (port # and dims)
    outport      = layer.find('output').find('port')
    outport_id   = int(outport.attrib['id'])
    outport_prec = outport.attrib['precision']
    outport_dims = outport.findall('dim')
    outport_dims_string = ""
    for dim in outport_dims:
        outport_dims_string += et.tostring(dim).decode('utf-8')

    # generate XML strings
    dummyLayer = """
        <layer id="9999" name="featuremap_checker_dummy_node" type="Result" version="opset1">
            <input>
                <port id="0">
                {}
                </port>
            </input>
        </layer>
    """.format(outport_dims_string)
    dummyEdge = '        <edge from-layer="{}" from-port="{}" to-layer="9999" to-port="0"/>'.format(nodeid, outport_id)

    # modify XML to make a dummy branch path for feature map extraction
    xmlcopy.find('layers').append(et.fromstring(dummyLayer))
    xmlcopy.find('edges').append(et.fromstring(dummyEdge))

    # return the modified XML and the name of the target node (specified by 'nodeid')
    return xmlcopy, layer.attrib['name']

def product(lis):
    if len(lis) == 0:
        return 0
    else:
        res = 1
        for x in lis:
            res *= x
        return res

def calc_rf(f, stride):
    rf = []
    for i in range(len(f)):
        if i == 0:
            rf.append(3)
        else:
            rf.append(rf[i-1] + (f[i]-1) * product(stride[:i]))
    return rf

def calc_l_star(template, rf):
    k = 3
    l = np.sum(rf <= min(list(template.shape[-2:]))) - 1
    l_star = max(l - k, 1)
    return l_star

def hook(target_layer, image, IE):
    for layer in layers.findall('layer'):
        nodeid = int(layer.attrib['id'])
        nodetype = layer.attrib['type']
        nodeName = layer.attrib['name']
        if nodetype in ['Const']: # , 'ShapeOf', 'Convert', 'StridedSlice', 'PriorBox']:
            continue
        if nodeName != names[target_layer]:
            continue
        if not layer.find('output') is None:
            outputport = layer.find('output').find('port')
            dims = []
            for dim in outputport.findall('dim'):                       # extract shape information
                dims.append(dim.text)

            modifiedXML, targetNodeName = modifyXMLForFeatureVectorProbing(originalXML, nodeid)
            XMLstr = et.tostring(modifiedXML.getroot())

            readNet_start = timeit.default_timer()
            net = IE.read_network(XMLstr, weight, init_from_buffer=True)
            readNet_end = timeit.default_timer()
            logging.info(f'Network Reading Time : {readNet_end - readNet_start}')

            # Reshape Feature
            input_blob = next(iter(net.inputs))

            n, c, h, w = net.input_info[input_blob].input_data.shape
            c_img, h_img, w_img  = image.shape
            net.reshape({input_blob: (n, c, h_img, w_img)})
            
            try:
                loadNet_start = timeit.default_timer()
                exenet = IE.load_network(net, DEVICE)
                loadNet_end = timeit.default_timer()
                logging.info(f'Network Loading Time : {loadNet_end - loadNet_start}')
            except RuntimeError:
                print('*** RuntimeError: load_network() -- Skip node \'{}\' - \'{}\''.format(targetNodeName, nodetype))
                continue
            
            ifer_start = timeit.default_timer()
            res = exenet.infer(inputs={input_blob: image})[nodeName]
            ifer_end = timeit.default_timer()
            logging.info(f'IE Inferencing Time : Image:({h_img,w_img}) -> {ifer_end - ifer_start}')

            del exenet
            del net
            
            return res

def calc_NCC(F, M):
    c, h_f, w_f = F.shape[-3:]
    tmp = np.zeros((c, M.shape[-2] - h_f, M.shape[-1] - w_f, h_f, w_f))
    for i in range(M.shape[-2] - h_f):
        for j in range(M.shape[-1] - w_f):
            M_tilde = M[:, :, i:i+h_f, j:j+w_f][:, None, None, :, :]
            tmp[:, i, j, :, :] = M_tilde / np.linalg.norm(M_tilde)
    NCC = np.sum(tmp*F.reshape(F.shape[-3], 1, 1, F.shape[-2], F.shape[-1]), axis=(0, 3, 4))
    return NCC

def nms(dets, scores, thresh):
    x1 = dets[:, 0, 0]
    y1 = dets[:, 0, 1]
    x2 = dets[:, 1, 0]
    y2 = dets[:, 1, 1]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep

if __name__ == '__main__':
    all_start = timeit.default_timer()
    originalXML = readXML(MODEL_PATH + 'custom_vgg16.xml')
    weight = readBIN(MODEL_PATH + 'custom_vgg16.xml')

    f = []
    strides = []
    names = []
    feature_vectors = {}

    root = originalXML.getroot()
    layers = root.find('layers')
    for layer in layers.findall('layer'):
        nodeName = layer.attrib['name']
        nodeid = int(layer.attrib['id'])
        nodetype = layer.attrib['type']
        
        if nodetype in ['Convolution']:
            const = layers[nodeid-1]
            kernel_size = const.find('data').attrib['shape'][-1]
            stride = layer.find('data').attrib['strides'][-1]
            f.append(int(kernel_size))
            strides.append(int(stride))
            names.append(nodeName)
        
        if nodetype in ['MaxPool']:
            kernel_size = layer.find('data').attrib['kernel'][-1]
            stride = layer.find('data').attrib['strides'][-1]
            f.append(int(kernel_size))
            strides.append(int(stride))
            names.append(nodeName)
        
        if nodetype in ['Const']: # , 'ShapeOf', 'Convert', 'StridedSlice', 'PriorBox']:
            #print(nodetype)
            continue

    rf = np.array(calc_rf(f, strides))

    sample_raw = cv2.imread('./data/sample1.jpg')
    sample_height, sample_width, sample_channels = sample_raw.shape
    sample = sample_raw.transpose((2, 0, 1))

    template_raw = cv2.imread('./data/template1.png')
    template_height, template_width, template_channels = template_raw.shape
    template = template_raw.transpose((2, 0, 1))

    l_star = calc_l_star(template, rf)
    print('Target Layer(L*):',l_star)
    
    # Inference
    ie_start = timeit.default_timer()
    IE = IECore()
    ie_end = timeit.default_timer()
    logging.info(f'IE Core Loading Time : {ie_end - ie_start}')
    
    fea_start = timeit.default_timer()
    F = hook(l_star, template, IE).astype(np.float32)
    M = hook(l_star, sample, IE).astype(np.float32)
   
    fea_stop = timeit.default_timer()
    print('Feature Extration Time ：', fea_stop - fea_start)
    ncc_start = timeit.default_timer()
    NCC = calc_NCC(F, M)
    ncc_stop = timeit.default_timer()
    print('NCC time : ', ncc_stop - ncc_start)

    threshold = 0.95 * np.max(NCC)
    max_indices = np.array(np.where(NCC > threshold)).T
    print("detected boxes: {}".format(len(max_indices)))

    template_feature_map = F
    image_feature_map = M

    sample_size = sample.shape
    template_size = template.shape

    boxes = []
    centers = []
    scores = []
    for max_index in max_indices:
        i_star, j_star = max_index
        NCC_part = NCC[i_star-1:i_star+2, j_star-2:j_star+2]
        
        
        x_center = (j_star + template_feature_map.shape
                    [-1]/2) * sample_size[-1] // image_feature_map.shape[-1]
        y_center = (i_star + template_feature_map.shape
                    [-2]/2) * sample_size[-2] // image_feature_map.shape[-2]
        
        x1_0 = x_center - template_size[-1]/2
        x2_0 = x_center + template_size[-1]/2
        y1_0 = y_center - template_size[-2]/2
        y2_0 = y_center + template_size[-2]/2

        stride_product = float(product(stride[:l_star]))
        
        x1 = np.sum(
            NCC_part * (x1_0 + np.array([-2, -1, 0, 1]) * stride_product)[None, :]) / np.sum(NCC_part)
        x2 = np.sum(
            NCC_part * (x2_0 + np.array([-2, -1, 0, 1]) * stride_product)[None, :]) / np.sum(NCC_part)
        y1 = np.sum(
            NCC_part * (y1_0 + np.array([-1, 0, 1]) * stride_product)[:, None]) / np.sum(NCC_part)
        y2 = np.sum(
            NCC_part * (y2_0 + np.array([-1, 0, 1]) * stride_product)[:, None]) / np.sum(NCC_part)

        x1 = int(round(x1))
        x2 = int(round(x2))
        y1 = int(round(y1))
        y2 = int(round(y2))
        x_center = int(round(x_center))
        y_center = int(round(y_center))

        boxes.append([(x1, y1), (x2, y2)])
        centers.append((x_center, y_center))
        scores.append(np.sum(NCC_part))

    nms_res = nms(np.array(boxes), np.array(scores), thresh=0.5)
    

    res_img = sample_raw.copy()
    res_img = cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB)

    for i in nms_res:
        res_img = cv2.rectangle(res_img, boxes[i][0], boxes[i][1], (255, 0, 0), 3)
        res_img = cv2.circle(res_img, centers[i], int(
            (boxes[i][1][0] - boxes[i][0][0])*0.2), (0, 0, 255), 2)
    all_stop = timeit.default_timer()

    print('All : ', all_stop - all_start)
    logging.info(f'Execution Time : {all_stop - all_start}')
    
    cv2.imwrite("RESULT.jpg", res_img)
