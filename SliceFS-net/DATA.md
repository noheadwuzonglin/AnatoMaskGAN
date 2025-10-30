# 数据集路径整理 总序列数量 8065

## 1. makeanything 序列数量 1050 (1000 for training, 50 for inference)

### condition_image
- `/opt/liblibai-models/user-workspace2/users/wgy/data/makeanything/frame4_split`
- `/opt/liblibai-models/user-workspace2/users/wgy/data/makeanything/frame9_split`

### 结果图
- `/opt/liblibai-models/user-workspace2/users/wgy/data/makeanything/frame4_split`
- `/opt/liblibai-models/user-workspace2/users/wgy/data/makeanything/frame9_split`

### prompt
- `/opt/liblibai-models/user-workspace2/users/wgy/data/makeanything/frame4_resolution1024_domains11_quantity11x50`
- `/opt/liblibai-models/user-workspace2/users/wgy/data/makeanything/frame9_resolution1056_domains10_quantity10x50`

> 具体prompt文件示例：  
> `/opt/liblibai-models/user-workspace2/users/wgy/data/makeanything/frame9_resolution1056_domains10_quantity10x50/Cook/03.caption`

---

## 2. 多人合照 序列数量 578 (550 for training, 28 for inference)

### 第一批（下载真实合照，自己剪裁人像id图）
#### condition_image
`/opt/liblibai-models/user-workspace2/users/wgy/code/Bagel/data/group_head_photo`

#### condition_image (cat)
`/opt/liblibai-models/user-workspace2/users/wgy/code/Bagel/data/gpt4o_group_cat_img`

#### 结果图
`/opt/liblibai-models/user-workspace2/users/wgy/code/Bagel/data/group_photo_backup`

#### prompt
```python
f"Create a professional group photo of {num_people} people from the provided individual portraits. Ensure consistent lighting, matching facial expressions (all smiling/natural), and seamless blending of backgrounds. Adjust heights and perspectives for a realistic look. The subjects should appear connected and engaged, as if they were photographed together."
```
### 第二批（下载ffhq数据集，随机挑选人像id图，送给gpt4o进行生成，爬虫获取）

#### condition_image
`/opt/liblibai-models/user-workspace2/users/wgy/code/Bagel/data/gpt4o_group_photo`

#### condition_image (cat)
`/opt/liblibai-models/user-workspace2/users/wgy/code/Bagel/data/gpt4o_group_cat_img`

#### 结果图
`/opt/liblibai-models/user-workspace2/users/wgy/code/Bagel/data/gpt4o_group_photo/*_generated.png`

#### prompt
JSON文件路径：  
`/opt/liblibai-models/user-workspace2/users/wgy/code/Bagel/data/gpt4o_group_photo/samples_index_0_520_work.json`

> prompt字段：`prompt_used`

---

## 3. 人和食品、店铺 序列数量 564 (536 for training, 28 for inference)

### 两种生成方式：

#### 方式一（人+食品+场景）
<img src="https://img.icons8.com/ios/20/000000/user.png"/> 人物来源：FFHQ数据集随机选择  
<img src="https://img.icons8.com/ios/20/000000/food.png"/> 食品来源：Yelp数据集  
<img src="https://img.icons8.com/ios/20/000000/room.png"/> 场景来源：Yelp数据集  
<img src="https://img.icons8.com/ios/20/000000/ai.png"/> 生成方式：GPT4O生成包含场景的人和食品图

#### 方式二（人+双食品）
<img src="https://img.icons8.com/ios/20/000000/user.png"/> 人物来源：FFHQ数据集随机选择  
<img src="https://img.icons8.com/ios/20/000000/food.png"/> <img src="https://img.icons8.com/ios/20/000000/food.png"/> 食品来源：Yelp数据集选择两张食品图  
<img src="https://img.icons8.com/ios/20/000000/ai.png"/> 生成方式：GPT4O生成人和两类食品图

### condition_image
`/opt/liblibai-models/user-workspace2/users/wgy/code/Bagel/data/gpt4o_product_img`

#### condition_image (cat)
`/opt/liblibai-models/user-workspace2/users/wgy/code/Bagel/data/gpt4o_product_cat_img_resize_four`

### 结果图
`/opt/liblibai-models/user-workspace2/users/wgy/code/Bagel/data/gpt4o_product_img/*_generated.png`

### prompt
JSON文件路径：  
`/opt/liblibai-models/user-workspace2/users/wgy/code/Bagel/data/gpt4o_product_img/samples_index_1_600_work.json`

> 使用字段：`prompt_used`

## 4. 人和商品 序列数量 5873
### condition_image
- `/opt/liblibai-models/user-workspace2/datasets/Multi_Control/input1`

### 结果图
- `/opt/liblibai-models/user-workspace2/datasets/Multi_Control/output`

### prompt
- `/opt/liblibai-models/user-workspace2/datasets/Multi_Control/caption`
