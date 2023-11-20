var g_httptype = 8;
function inithttp() {
	var xmlHttp  = "";
	if (window.XMLHttpRequest) {
		xmlHttp = new XMLHttpRequest();
		if (xmlHttp.overrideMimeType) {
			xmlHttp.overrideMimeType("text/xml");
		}
		g_httptype = 8;
	} else if (window.ActiveXObject) {
		var activexName = ['MSXML2.XMLHTTP.6.0', 'MSXML2.XMLHTTP.5.0', 'MSXML2.XMLHTTP.4.0', 'MSXML2.xmlhttp.3.0', 'MSXML2.XMLHTTP.2.0', 'MSXML2.XMLHTTP.1.0',
						'MSXML2.XMLHTTP', 'Microsoft.XMLHTTP'];
		for(var i = 0; i < activexName.length; i++) {
			try {
				xmlHttp = new ActiveXObject(activexName[i]);
				g_httptype = i;
				break;
			} catch(e) {
			}
		}
	}
	return xmlHttp;
}
				
function orfldcfg() {
	var obj = document.getElementsByName("orfldsel");
	var check_val = [];
	var clen = obj.length;
	if(clen < 7) {
		alert("OR Filed Item number error!");
		return;
	}
	for (var i = 0; i < clen; i++) {
		if (obj[i].checked) {
			check_val.push(obj[i].value);
		} else {
			check_val.push(0);
		}
	}
	var s_orbttbl = "<?xml version=\"1.0\" encoding=\"UTF-8\"?><root><APP><ENVI name=\"应用环境\"></ENVI><BUSI name=\"业务名称\"><INDU name=\"行业种类\"></INDU></BUSI><COND name=\"运行条件\"></COND></APP><REQ><NEED name=\"需求描述\"></NEED><WANT name=\"诉求描述\"></WANT></REQ>";
	
	s_orjson = [
		{A: '编号',B: '字段', C: '子字段', D: '选择'},
		{A: 'APP', B: '场景', C: '应用环境', D: '必选'},
		{A: 'APP', B: '', C: '业务名称', D: '必选'},
		{A: 'APP', B: '', C: '行业种类', D: '可选'},
		{A: 'APP', B: '', C: '运行条件', D: '可选'},
		{A: 'REQ', B: '需求', C: '需求说明', D: '必选'},
		{A: 'REQ', B: '', C: '诉求说明', D: '可选'}];
	if(check_val[2]==3) {
		s_orbttbl = s_orbttbl + "<GAP><COMP name=\"竞争对手\"></COMP><GBUS name=\"差距业务\"></GBUS><GDES name=\"差距需求\"></GDES></GAP>";
		s_orjson = s_orjson.concat([{A: 'GAP', B: '差距', C: '竞争对手', D: '必选'},
		{A: 'GAP', B: '', C: '差距业务', D: '必选'},
		{A: 'GAP', B: '', C: '差距需求', D: '可选'}]);
	}
	if(check_val[3]==4) {
		s_orbttbl = s_orbttbl + "<PRO><CUST name=\"痛点客户\"></CUST><PBUS name=\"痛点业务\"></PBUS><PDES name=\"痛点需求\"></PDES></PRO>";
		s_orjson = s_orjson.concat([{A: 'PRO', B: '痛点', C: '痛点客户', D: '必选'},
		{A: 'PRO', B: '', C: '痛点业务', D: '必选'},
		{A: 'PRO', B: '', C: '痛点需求', D: '可选'}]);
	}
	if(check_val[4]==5) {
		s_orbttbl = s_orbttbl + "<VAL><VCUS name=\"客户价值\"></VCUS><VINT name=\"内部价值\"></VINT></VAL>";
		s_orjson = s_orjson.concat([{A: 'VAL', B: '价值', C: '客户价值', D: '必选'},
		{A: 'VAL', B: '', C: '内部价值', D: '可选'}]);
	}
	if(check_val[5]==6) {
		s_orbttbl = s_orbttbl + "<CHK><CDAT name=\"标准要求\"></CDAT><CDES name=\"验收日期\"></CDES></CHK>";
		s_orjson = s_orjson.concat([{A: 'CHK', B: '验收标准', C: '标准要求', D: '必选'},
		{A: 'CHK', B: '', C: '验收日期', D: '必选'}]);
	}
	if(check_val[6]==7) {
		s_orbttbl = s_orbttbl + "<DEP><DOUT name=\"外部依赖\"></DOUT><CINT name=\"内部依赖\"></CNT><DANR name=\"场景关联\"></DANR><DVNR name=\"价值关联\"></DVNR></DEP></root>";
		s_orjson = s_orjson.concat([{A: 'DEP', B: '依赖关系', C: '外部依赖', D: '必选'},
		{A: 'DEP', B: '', C: '内部依赖', D: '必选'},
		{A: 'DEP', B: '', C: '场景关联', D: '可选'},
		{A: 'DEP', B: '', C: '价值关联', D: '可选'}]);
	}
	var workbook  = XLSX.utils.book_new();
    var worksheet = XLSX.utils.json_to_sheet(s_orjson);
    //var blob = new Blob([s2ab(s_orbttbl)], { type: "application/octet-stream" });
    //var xmlDoc = loadXMLDoc("ORDefTbl.xml");
    //var x2js = new X2JS();
    //var jsonObj = x2js.xml2json(xmlDoc);
    //var s_orbtxml = JSON.stringify(jsonObj);
    //alert(s_orbtxml);
    //var worksheet2 = XLSX.utils.json_to_sheet(s_orbtxml);
    XLSX.utils.book_append_sheet(workbook, worksheet, "ORFldItem");
    //XLSX.utils.book_append_sheet(workbook, worksheet2, "Sheet2");
	//XLSX.writeFile(workbook, "ORFldTbl.xlsx", { compression: true });
	//alert(s_orbttbl);
	alert("OR Field Item File Generate To Download!");
	var wopts = { bookType: 'xlsx', bookSST: false, type: 'binary' };
	var wbout = XLSX.write(workbook, wopts);
	var blob = new Blob([s2ab(wbout)], { type: "application/octet-stream" });
	saveAs(blob, "ORFldTbl.xlsx");
	var blob = new Blob([s2ab(s_orbttbl)], { type: "application/octet-stream" });
	saveAs(blob, "ORFldTbl.xml");
	return;
}
function s2ab(s) {
 	var buf = new ArrayBuffer(s.length);
	var view = new Uint8Array(buf);
	for (var i=0; i<s.length; i++) view[i] = s.charCodeAt(i) & 0xFF;
	return buf;
}
function saveAs(obj, filename) {
	var link = document.createElement("a");
    link.download = filename;
    link.href = URL.createObjectURL(obj);
    link.click();
    URL.revokeObjectURL(obj);
}
function loadXMLDoc(dname) {
    if (window.XMLHttpRequest) {
        xhttp=new XMLHttpRequest();
    }
    else {
        xhttp=new ActiveXObject("Microsoft.XMLHTTP");
    }
    xhttp.open("GET",dname,false);
    xhttp.send();
    return xhttp.responseXML;
}
function orchkcfg() {
	var obj = document.getElementsByName("orchksel");
	var check_val = [];
	var clen = obj.length;
	if(clen < 3) {
		alert("OR Check Rules number error!");
		return;
	}
	for (var i = 0; i < clen; i++) {
		if (obj[i].checked) {
			check_val.push(obj[i].value);
		} else {
			check_val.push(0);
		}
	}
	return;
}

function aigor() {
	getproccfg();
	/*var rspval = "901";
	var xmlHttp = inithttp();
	if(xmlHttp == undefined || xmlHttp == null) {
		alert("对不起, 您的浏览器不支持HTTP请求对象!");
		return;
	}
	xmlHttp.open("POST", "aigor.asp?biaozhu=1&tiaozheng=2&jiaozheng=3", false);
	if (g_httptype != 8) {
		xmlhttp.setRequestHeader("Content-Type","application/x-www-form-urlencoded");
	}
	xmlHttp.onreadystatechange = function () {
	    if (xmlHttp.readyState == 4) {
	        if (xmlHttp.status == 200) {
	            rspval = xmlHttp.responseText;
	        	alert("-----------"+rspval);
			} else {
	            alert("error status-"+xmlHttp.status);
	        	return;
	        }
	    }
	}
	xmlHttp.send(null);

	if (rspval == "901") {
		alert("sorry, can not connect server!");
	} else {
		alert("success!");
	}*/
	return;
}
function getproccfg() {
	var obj = document.getElementsByName("aigselect");
	var check_val = [];
	var proc_val  = ["labz_sel", "ajfld_sel", "ajchk_sel"];
	var clen = obj.length;
	if(clen < 3) {
		alert("OR Byte length error!");
		return;
	}
	var s_orjson = [];
	for (var i = 0; i < clen; i++) {
		if (obj[i].checked) {
			check_val.push(obj[i].value);
			s_orjson = s_orjson.concat([{A: proc_val[i], B: '1'}]);
		} else {
			check_val.push(0);
			s_orjson = s_orjson.concat([{A: proc_val[i], B: '0'}]);
		}
	}
	var workbook  = XLSX.utils.book_new();
    var worksheet = XLSX.utils.json_to_sheet(s_orjson);
	XLSX.utils.book_append_sheet(workbook, worksheet, "ProcCfg");
	alert("AIGOR Process CFG File Generate To Download!");
	var wopts = { bookType: 'xlsx', bookSST: false, type: 'binary' };
	var wbout = XLSX.write(workbook, wopts);
	var blob = new Blob([s2ab(wbout)], { type: "application/octet-stream" });
	saveAs(blob, "ProcCfg.xlsx");
	return;
}
function orcpssort() {
	var obj = document.getElementsByName("aigselect");
	
	return;
}
