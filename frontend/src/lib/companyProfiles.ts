export interface CompanyProfile {
  symbol: string;
  name: string;
  sector: string;
  industry?: string;
  description?: string;
  website?: string;
}

export const COMPANY_PROFILES: Record<string, CompanyProfile> = {
  ALI: {
    symbol: "ALI",
    name: "Ayala Land, Inc.",
    sector: "Property",
    industry: "Real Estate Development",
    description:
      "Ayala Land, Inc. is the largest real estate developer in the Philippines, engaged in master-planned mixed-use estates, residential developments, shopping centers, commercial offices, hotels, and resorts.",
    website: "https://www.ayalaland.com.ph",
  },
  APX: {
    symbol: "APX",
    name: "Apex Mining Co., Inc.",
    sector: "Mining and Oil",
    industry: "Gold & Silver Mining",
    description:
      "Apex Mining Co., Inc. is a Philippine mineral exploration and mining company operating the Maco Gold Mine in Maco, Davao de Oro and the Sangilo Mine in Itogon, Benguet.",
    website: "https://www.apexmining.com",
  },
  BPI: {
    symbol: "BPI",
    name: "Bank of the Philippine Islands",
    sector: "Financials",
    industry: "Universal Commercial Banking",
    description:
      "Bank of the Philippine Islands is the first bank in the Philippines and Southeast Asia, offering universal commercial banking, consumer lending, asset management, and investment banking services.",
    website: "https://www.bpi.com.ph",
  },
  GLO: {
    symbol: "GLO",
    name: "Globe Telecom, Inc.",
    sector: "Services",
    industry: "Telecommunications",
    description:
      "Globe Telecom, Inc. is a leading Philippine telecommunications and digital infrastructure company providing nationwide mobile voice, data, fixed-line broadband, and digital financial solutions.",
    website: "https://www.globe.com.ph",
  },
  ICT: {
    symbol: "ICT",
    name: "Intl. Container Terminal Services",
    sector: "Services",
    industry: "Port Operations & Logistics",
    description:
      "International Container Terminal Services, Inc. (ICTSI) is an international terminal operator headquartered in Manila, acquiring, developing, and operating container ports across the globe.",
    website: "https://www.ictsi.com",
  },
  JFC: {
    symbol: "JFC",
    name: "Jollibee Foods Corporation",
    sector: "Industrial",
    industry: "Food & Quick Service Restaurants",
    description:
      "Jollibee Foods Corporation is one of the world's largest and fastest-growing Asian restaurant companies, operating flagship brand Jollibee alongside a multinational portfolio of quick-service food chains.",
    website: "https://www.jollibee.com.ph",
  },
  MBT: {
    symbol: "MBT",
    name: "Metropolitan Bank & Trust Co.",
    sector: "Financials",
    industry: "Universal Commercial Banking",
    description:
      "Metropolitan Bank & Trust Company (Metrobank) is a premier universal bank in the Philippines, delivering corporate banking, consumer loans, investment banking, and treasury services.",
    website: "https://www.metrobank.com.ph",
  },
  MEG: {
    symbol: "MEG",
    name: "Megaworld Corporation",
    sector: "Property",
    industry: "Real Estate & Urban Townships",
    description:
      "Megaworld Corporation is a leading real estate developer and pioneer of the 'live-work-play-learn' integrated urban township model in the Philippines.",
    website: "https://www.megaworldcorp.com",
  },
  MER: {
    symbol: "MER",
    name: "Manila Electric Company",
    sector: "Industrial",
    industry: "Electric Power Distribution",
    description:
      "Manila Electric Company (Meralco) is the largest electric power distribution company in the Philippines, serving Metro Manila and surrounding economic corridor provinces.",
    website: "https://www.meralco.com.ph",
  },
  NIKL: {
    symbol: "NIKL",
    name: "Nickel Asia Corporation",
    sector: "Mining and Oil",
    industry: "Nickel Ore Mining & Processing",
    description:
      "Nickel Asia Corporation is the Philippines' largest producer of lateritic nickel ore and one of the largest in the world, exporting saprolite and limonite ore and investing in renewable power projects.",
    website: "https://www.nickelasia.com",
  },
  PGOLD: {
    symbol: "PGOLD",
    name: "Puregold Price Club, Inc.",
    sector: "Services",
    industry: "Retail Grocery & Hypermarkets",
    description:
      "Puregold Price Club, Inc. operates a nationwide retail chain of hypermarkets, supermarkets, and wholesale clubs catering to retail consumers and small grocery enterprises (sari-sari stores).",
    website: "https://www.puregold.com.ph",
  },
  SCC: {
    symbol: "SCC",
    name: "Semirara Mining and Power Corp.",
    sector: "Mining and Oil",
    industry: "Coal Mining & Thermal Power",
    description:
      "Semirara Mining and Power Corporation is the only large-scale sub-bituminous coal producer in the Philippines, integrating open-pit coal mining in Caluya, Antique with thermal power generation.",
    website: "https://www.semiraramining.com",
  },
  SECB: {
    symbol: "SECB",
    name: "Security Bank Corporation",
    sector: "Financials",
    industry: "Commercial & Retail Banking",
    description:
      "Security Bank Corporation is an independent universal bank in the Philippines, offering commercial banking, consumer loans, treasury, and wealth management services.",
    website: "https://www.securitybank.com",
  },
  SHLPH: {
    symbol: "SHLPH",
    name: "Pilipinas Shell Petroleum Corp.",
    sector: "Industrial",
    industry: "Petroleum & Energy Distribution",
    description:
      "Shell Pilipinas Corporation (formerly Pilipinas Shell Petroleum Corp.) is an energy enterprise engaged in the importation, distribution, and marketing of petroleum fuels, lubricants, and mobility services.",
    website: "https://www.shell.com.ph",
  },
  SMPH: {
    symbol: "SMPH",
    name: "SM Prime Holdings, Inc.",
    sector: "Property",
    industry: "Real Estate & Mall Development",
    description:
      "SM Prime Holdings, Inc. is one of the largest integrated property developers in Southeast Asia, developing and operating SM shopping malls, residential condominiums, commercial offices, and hotels.",
    website: "https://www.smprime.com",
  },
};

export function getCompanyProfile(symbol: string): CompanyProfile | undefined {
  return COMPANY_PROFILES[symbol.toUpperCase()];
}
