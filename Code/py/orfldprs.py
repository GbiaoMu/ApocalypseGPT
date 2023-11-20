# _*_coding:utf-8_*_
import os, sys, string
import pandas as pd
import openpyxl

def fldprs():
	fldxl = pd.read_excel(r'./ORFldTbl1.xlsx', 'ORFldItem')
	pmptlst = []
	fname = "sortprompt.txt"
	ifile = open(fname, mode='w')
	i = 0
	for vala,valb,valc,vald in zip(fldxl.A.values,fldxl.B.values,fldxl.C.values,fldxl.D.values):
		fldbstr = ""
		fldcstr = ""
		if (vala == "APP") and (valb == "场景") and (vald == "必选"):
			fldbstr = "这段话或这句话是说明应用场景或客户如何使用吗?\n"
#			fldcstr = "针对应用场景或客户使用场景，这段话或这句话是说明%s的吗?\n"%(valc)
		if (vala == "APP") and (vald == "必选"):
			fldcstr = "针对应用场景或客户使用场景，这段话或这句话是说明%s的吗?\n"%(valc)
			
		if (vala == "REQ") and (valb == "需求") and (vald == "必选"):
			fldbstr = "这段话或这句话是说明应用需求或客户需求吗?\n"
#			fldcstr = "针对应用需求或客户需求，这段话或这句话是说明客户%s的吗?\n"%(valc)
		if (vala == "REQ") and (vald == "必选"):
			fldcstr = "针对应用需求或客户需求，这段话或这句话是说明客户%s的吗?\n"%(valc)
			
		if (vala == "GAP") and (valb == "差距") and (vald == "必选"):
			fldbstr = "这段话或这句话是说明自身与竞争对手产品的差距吗?\n"
#			fldcstr = "针对与竞争对手产品的差距，这段话或这句话表明的%s是什么?\n"%(valc)
		if (vala == "GAP") and (vald == "必选"):
			fldcstr = "针对与竞争对手产品的差距，这段话或这句话表明的%s是什么?\n"%(valc)
		
		if (vala == "PRO") and (valb == "痛点") and (vald == "必选"):
			fldbstr = "这段话或这句话是说明客户痛点或客户应用的问题吗?\n"
#			fldcstr = "针对客户痛点或客户应用的问题，这段话或这句话表明的%s是什么?\n"%(valc)
		if (vala == "PRO") and (vald == "必选"):
			fldcstr = "针对客户痛点或客户应用的问题，这段话或这句话表明的%s是什么?\n"%(valc)
			
		if (vala == "VAL") and (valb == "价值") and (vald == "必选"):
			fldbstr = "这段话或这句话是说明产品带给客户的价值或自身的价值吗?\n"
#			fldcstr = "针对产品带给客户的价值或自身的价值，这段话或这句话是说明%s的吗?\n"%(valc)
		if (vala == "VAL") and (vald == "必选"):
			fldcstr = "针对产品带给客户的价值或自身的价值，这段话或这句话是说明%s的吗?\n"%(valc)
			
		if (vala == "CHK") and (valb == "验收标准") and (vald == "必选"):
			fldbstr = "这段话或这句话是说明客户关于产品验收标准相关内容的吗?\n"
#			fldcstr = "针对客户关于产品验收标准相关内容，这段话或这句话表明的%s是什么?\n"%(valc)
		if (vala == "CHK") and (vald == "必选"):
			fldcstr = "针对客户关于产品验收标准相关内容，这段话或这句话表明的%s是什么?\n"%(valc)
			
		if (vala == "DEP") and (valb == "依赖关系") and (vald == "必选"):
			fldbstr = "这段话或这句话是说明需求对内或对外依赖关系或需求的关联关系吗?\n"
#			fldcstr = "针对需求对内或对外依赖关系或需求的关联关系，这段话或这句话表明的%s是什么?\n"%(valc)
		if (vala == "DEP") and (vald == "必选"):
			fldcstr = "针对需求对内或对外依赖关系或需求的关联关系，这段话或这句话表明的%s是什么?\n"%(valc)
		pmptlst.append(fldbstr)
		pmptlst.append(fldcstr)
		ifile.write(fldbstr)
		ifile.write(fldcstr)
	ifile.close()
#	print(pmptlst)
	
def fldprs():
	
if __name__ == '__main__':
    fldprs()