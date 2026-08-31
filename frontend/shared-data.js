/**
 * 智慧水务系统 - 全局数据共享模块
 * 存储格式: { fileName, data: [...], lastUpdated }
 */

const STORAGE_KEY = 'waterData';

const AquaMindData = {
    // 获取完整存储对象
    getStorage() {
        try {
            const data = localStorage.getItem(STORAGE_KEY);
            return data ? JSON.parse(data) : null;
        } catch (e) {
            console.error('读取数据失败:', e);
            return null;
        }
    },

    // 获取数据数组
    getData() {
        const storage = this.getStorage();
        return storage ? storage.data : null;
    },

    // 获取元数据
    getMeta() {
        const storage = this.getStorage();
        if (!storage) return null;
        return {
            fileName: storage.fileName,
            uploadTime: storage.lastUpdated,
            recordCount: storage.data ? storage.data.length : 0
        };
    },

    // 保存数据
    saveData(data, fileName) {
        try {
            const storageObj = {
                fileName: fileName,
                data: data,
                lastUpdated: new Date().toISOString()
            };
            localStorage.setItem(STORAGE_KEY, JSON.stringify(storageObj));
            console.log('✅ 数据已保存:', fileName, data.length, '条');
            return true;
        } catch (e) {
            console.error('保存失败:', e);
            return false;
        }
    },

    // 清空数据
    clearData() {
        localStorage.removeItem(STORAGE_KEY);
    },

    // 检查是否有数据
    hasData() {
        const storage = this.getStorage();
        return storage !== null && storage.data && storage.data.length > 0;
    },

    // 识别列名
    detectColumns(headers) {
        const map = {};
        headers.forEach(h => {
            const lower = h.toLowerCase();
            // 时间
            if (/时间|time|timestamp|日期|datetime|date/i.test(h)) map.timestamp = h;
            // 浊度（进水/原水）
            else if (/^(?!.*出水|.*effluent).*浊度|^(?!.*effluent).*turbidity|ntu|influent turbidity/i.test(h)) map.turbidity = h;
            // 出水浊度
            else if (/出水浊度|effluent turbidity/i.test(h)) map.effluentTurbidity = h;
            // pH
            else if (/^ph$|^pH$/i.test(h)) map.ph = h;
            // 余氯
            else if (/余氯|chlorine/i.test(h)) map.chlorine = h;
            // COD
            else if (/cod|化学需氧量/i.test(lower)) map.cod = h;
            // 氨氮
            else if (/氨氮|ammonia|nh3|ammonia n/i.test(h)) map.ammonia = h;
            // 溶解氧
            else if (/溶解氧|do/i.test(h)) map.do = h;
            // 电导率
            else if (/电导率|conductivity/i.test(h)) map.conductivity = h;
            // 加药量/混凝剂
            else if (/加药量|dosage|dose|加药|coagulant|pac/i.test(h)) map.dosage = h;
            // 设备健康度
            else if (/健康度|health|device_health/i.test(h)) map.health = h;
            // 能耗/功率
            else if (/能耗|energy|power|用电/i.test(h)) map.power = h;
            // 温度
            else if (/温度|temperature|temp|水温/i.test(h)) map.temperature = h;
            // 流量/进水量
            else if (/流量|flow|inflow|进水|influent/i.test(h)) map.inflow = h;
            // 出水量
            else if (/出水量|outflow|effluent flow/i.test(h)) map.outflow = h;
            // 臭氧
            else if (/臭氧|ozone|pre.ozone|post.ozone/i.test(h)) map.ozone = h;
            // 砂滤池相关
            else if (/砂滤|sand filter/i.test(h)) map.sandFilter = h;
            // 反冲洗
            else if (/反冲洗|backwash/i.test(h)) map.backwash = h;
            // 沉降/沉淀
            else if (/沉降|沉淀|sedimentation/i.test(h)) map.sedimentation = h;
            // 矾花
            else if (/矾花|floc/i.test(h)) map.floc = h;
        });
        return map;
    },

    // 智能解析CSV - 支持复杂格式
    parseCSV(file, onSuccess, onError) {
        if (typeof Papa === 'undefined') {
            if (onError) onError('PapaParse 未加载');
            return;
        }

        // 先读取原始内容分析结构
        Papa.parse(file, {
            header: false,
            skipEmptyLines: false,
            complete: (preview) => {
                const rawData = preview.data;
                
                // 自动检测表头行（找包含关键词的行）
                let headerRowIndex = 0;
                let maxScore = 0;
                
                for (let i = 0; i < Math.min(10, rawData.length); i++) {
                    const row = rawData[i];
                    if (!row || row.length === 0) continue;
                    
                    const rowText = row.join(' ').toLowerCase();
                    let score = 0;
                    
                    // 检测时间相关关键词
                    if (/date|time|month|day|year/i.test(rowText)) score += 5;
                    // 检测水质指标关键词
                    if (/turbidity|ph|flow|tss|cod|bod|ammonia|chlorine/i.test(rowText)) score += 3;
                    // 检测进出水相关
                    if (/influent|effluent|infl|effl|actual/i.test(rowText)) score += 2;
                    // 检测是数据行（数字多）还是表头行
                    const numericCount = row.filter(cell => /\d/.test(cell)).length;
                    if (numericCount < 2) score += 2; // 数字少更可能是表头
                    
                    if (score > maxScore) {
                        maxScore = score;
                        headerRowIndex = i;
                    }
                }
                
                // 重新解析，使用检测到的表头行
                Papa.parse(file, {
                    header: true,
                    skipEmptyLines: true,
                    transformHeader: (header) => {
                        // 清理列名
                        return header.trim().replace(/\s+/g, ' ').replace(/^["']|["']$/g, '');
                    },
                    complete: (results) => {
                        if (results.errors.length > 0 && results.data.length === 0) {
                            if (onError) onError('CSV解析错误: ' + results.errors[0].message);
                            return;
                        }
                        
                        let data = results.data;
                        if (data.length === 0) {
                            if (onError) onError('CSV为空');
                            return;
                        }
                        
                        // 清理数据行
                        data = data.filter(row => {
                            // 过滤掉空行
                            if (!row || Object.keys(row).length === 0) return false;
                            // 过滤掉汇总行
                            const firstValue = Object.values(row)[0];
                            if (typeof firstValue === 'string' && 
                                /^(total|weighted|average|sum|note|updated|reviewed)/i.test(firstValue)) return false;
                            // 过滤掉公式行
                            if (typeof firstValue === 'string' && firstValue.includes('#')) return false;
                            // 至少有一个有效数值
                            const hasNumber = Object.values(row).some(v => /\d/.test(String(v)));
                            return hasNumber;
                        });
                        
                        if (data.length === 0) {
                            if (onError) onError('没有找到有效数据行');
                            return;
                        }
                        
                        // 获取并清理表头
                        let headers = Object.keys(data[0]).map(h => h.trim());
                        
                        // 如果没有检测到标准列名，尝试智能匹配
                        const columnMap = this.detectColumns(headers);
                        
                        // 如果没有时间列，尝试用第一列作为时间/序号列
                        if (!columnMap.timestamp && headers.length > 0) {
                            // 检查第一列是否包含月份/日期信息
                            const firstCol = headers[0];
                            const sampleValues = data.slice(0, 3).map(row => row[firstCol]);
                            const looksLikeDate = sampleValues.some(v => /\d{1,4}[\/\-\.]\d{1,2}|jan|feb|mar|apr|may|jun|jul/i.test(String(v)));
                            if (looksLikeDate) {
                                columnMap.timestamp = firstCol;
                            }
                        }
                        
                        // 查找浊度相关列（进水和出水）
                        headers.forEach(h => {
                            const lowerH = h.toLowerCase();
                            // 进水浊度
                            if (!columnMap.turbidity && 
                                (/(influent|infl).*turbidity/i.test(h) || 
                                 /turbidity.*(influent|infl)/i.test(h) ||
                                 /^tss$|^total suspended solids$/i.test(h))) {
                                columnMap.turbidity = h;
                            }
                            // 出水浊度
                            if (!columnMap.effluentTurbidity && 
                                /(effluent|effl).*turbidity/i.test(h)) {
                                columnMap.effluentTurbidity = h;
                            }
                            // 流量
                            if (!columnMap.inflow && 
                                /^(flow|inflow|influent).*(mgd|m3\/d)?$/i.test(h)) {
                                columnMap.inflow = h;
                            }
                            // TSS/SS
                            if (!columnMap.tss && 
                                /^(tss|total suspended solids|suspended solids)$/i.test(h)) {
                                columnMap.tss = h;
                            }
                            // COD
                            if (!columnMap.cod && 
                                /^(cod|cbod|carbonaceous).*demand/i.test(h)) {
                                columnMap.cod = h;
                            }
                            // BOD
                            if (!columnMap.bod && 
                                /^(bod|biochemical).*demand/i.test(h)) {
                                columnMap.bod = h;
                            }
                            // 去除率
                            if (!columnMap.removal && 
                                /% removal|removal.*%|percent removal/i.test(h)) {
                                columnMap.removal = h;
                            }
                        });
                        
                        // 校验 - 至少需要有一个可识别的指标
                        const hasAnyMetric = Object.keys(columnMap).length > 0;
                        
                        if (!hasAnyMetric) {
                            console.warn('可用列:', headers.join(', '));
                            if (onError) onError('未能识别CSV列名。支持的列名包括：Date, Flow, Turbidity, TSS, COD, BOD, pH等');
                            return;
                        }
                        
                        this.saveData(data, file.name);
                        
                        if (onSuccess) {
                            onSuccess({
                                data: data,
                                columnMap: columnMap,
                                fileName: file.name,
                                recordCount: data.length,
                                detectedHeaders: headers
                            });
                        }
                    },
                    error: (error) => {
                        if (onError) onError('文件读取失败: ' + error.message);
                    }
                });
            }
        });
    }
};

window.AquaMindData = AquaMindData;

// 页面加载后触发数据就绪事件
document.addEventListener('DOMContentLoaded', function() {
    if (AquaMindData.hasData()) {
        const data = AquaMindData.getData();
        const meta = AquaMindData.getMeta();
        window.dispatchEvent(new CustomEvent('aquamind:data-ready', {
            detail: {
                data: data,
                meta: meta,
                columnMap: AquaMindData.detectColumns(Object.keys(data[0]))
            }
        }));
    }
});

// 监听其他页面数据更新
window.addEventListener('storage', function(e) {
    if (e.key === STORAGE_KEY) {
        const storage = AquaMindData.getStorage();
        if (storage) {
            window.dispatchEvent(new CustomEvent('aquamind:data-updated', {
                detail: {
                    data: storage.data,
                    meta: {
                        fileName: storage.fileName,
                        uploadTime: storage.lastUpdated,
                        recordCount: storage.data.length
                    },
                    columnMap: AquaMindData.detectColumns(Object.keys(storage.data[0]))
                }
            }));
        }
    }
});

console.log('✅ AquaMindData 已加载');
